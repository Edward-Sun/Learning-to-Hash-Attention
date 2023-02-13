# Lint as: python3
"""Attention modules for Reformer model."""

import functools

import jax
import jax.numpy as jnp
from absl import logging
from flax import nn
from jax.scipy.special import logsumexp


def length_normalized(x, epsilon=1e-6):
  norm_inputs = nn.LayerNorm(x, epsilon=epsilon, bias=False, scale=False)
  return norm_inputs


def look_one_back(x):
  """Looks back to previous chunk.

  Args:
    x: input tensor of shape [num_chunks, div_len, dim]
  Returns:
    output tensor of shape [num_chunks, div_len * 2, dim]
  """
  xlb = jnp.concatenate([x[-1:, ...], x[:-1, ...]], axis=0)
  return jnp.concatenate([x, xlb], axis=1)


def look_one_back_and_ahead(x):
  """Looks back to previous chunk.

  Args:
    x: input tensor of shape [num_chunks, div_len, dim]
  Returns:
    output tensor of shape [num_chunks, div_len * 2, dim]
  """
  xlb = jnp.concatenate([x[-1:, ...], x[:-1, ...]], axis=0)
  xla = jnp.concatenate([x[1:, ...], x[:1, ...]], axis=0)
  return jnp.concatenate([x, xlb, xla], axis=1)


def permute_via_gather(val, permutation, inverse_permutation, axis=0):
  """Permutation helper for LSH attention."""
  # It is *not* safe to use jax.custom_vjp here. The most likely cause is that
  # it can't close over values: https://github.com/google/jax/issues/2676
  # The error only occurs in some configurations (e.g. use_python_loop = True,
  # num_parallel_heads = 1) but not others.
  permutation = jax.lax.stop_gradient(permutation)
  inverse_permutation = jax.lax.stop_gradient(inverse_permutation)

  def permute_impl(val):
    return jnp.take(val, permutation, axis=axis)

  def permute_vjp(val):
    permuted = permute_impl(jax.lax.stop_gradient(val))

    def vjpfun(permuted_grad):
      # JAX autodiff would synthesize a scatter operation because it doesn't
      # know that the indices are a permutatation. However on TPU, gathers are
      # faster than scatters (at least in the regime the LSH attention uses).
      return jnp.take(permuted_grad, inverse_permutation, axis=axis),

    return permuted, vjpfun

  permute = jax.custom_transforms(permute_impl)
  jax.defvjp_all(permute, permute_vjp)
  return permute(val)


def permute_via_sort(val, keys, inverse_keys, axis=0):
  """Permutation helper for LSH attention."""
  # It is *not* safe to use jax.custom_vjp here (see permute_via_gather).
  keys = jax.lax.stop_gradient(keys)
  inverse_keys = jax.lax.stop_gradient(inverse_keys)

  def permute_impl(val):
    # On TPU, sorting scalars by key is faster than a gather.
    _, permuted = jax.lax.sort_key_val(keys, val, dimension=axis)
    return permuted

  def permute_vjp(val):
    permuted = permute_impl(jax.lax.stop_gradient(val))

    def vjpfun(permuted_grad):
      _, val_grad = jax.lax.sort_key_val(
          inverse_keys, permuted_grad, dimension=axis)
      return val_grad,

    return permuted, vjpfun

  permute = jax.custom_transforms(permute_impl)
  jax.defvjp_all(permute, permute_vjp)
  return permute(val)


def hash_vectors(vecs, rng, num_buckets, num_hashes):
  """Performs batched hashing.

  Args:
    vecs: input of [length, dim].
    rng: rng object.
    num_buckets: integer, number of buckets.
    num_hashes: integer, number of hashes.
  Returns:
    output of shape [batch_size, length]
  """

  # batch_size = vecs.shape[0]

  assert num_buckets % 2 == 0

  rot_size = num_buckets

  rotations_shape = (vecs.shape[-1], num_hashes, rot_size // 2)

  rng = jax.lax.stop_gradient(jax.lax.tie_in(vecs, rng))
  random_rotations = jax.random.normal(rng, rotations_shape).astype(jnp.float32)

  rotated_vecs = jnp.einsum('tf,fhi->hti', vecs, random_rotations)
  rotated_vecs = jnp.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
  buckets = jnp.argmax(rotated_vecs, axis=-1)
  # [num_hashes, length]

  offsets = jax.lax.tie_in(buckets, jnp.arange(num_hashes))
  offsets = jnp.reshape(offsets * num_buckets, (-1, 1))
  buckets = jnp.reshape(buckets + offsets, (-1,))

  return buckets


def lsh_attention_single_batch(query, key, value, padding_mask,
                               n_buckets, n_hashes,
                               self_mask=False,
                               padding_masked=False,
                               look_both_side=True,
                               causal_mask=False):
  """LSH attention for single batch."""
  attn = jax.vmap(lsh_attention_single_head,
                  in_axes=(1, 1, 1,
                           None, None, None,
                           None, None, None, None),
                  out_axes=1)

  out = attn(query, key, value,
             padding_mask, n_buckets, n_hashes,
             self_mask, padding_masked, look_both_side, causal_mask)
  return out


def lsh_attention_single_head(query, key, value,
                              padding_mask, n_buckets, n_hashes,
                              self_mask=False,
                              padding_masked=False,
                              look_both_side=False,
                              causal_mask=False):
  """Applies LSH attention on a single head and a single batch."""
  qdim, vdim = query.shape[-1], value.shape[-1]
  chunk_size = n_hashes * n_buckets

  seqlen = query.shape[0]
  rng = nn.make_rng()
  total_hashes = n_hashes

  def get_sorted_array(array, rng):
    buckets = hash_vectors(
        array, rng, num_buckets=n_buckets, num_hashes=n_hashes)
    # buckets should be (seq_len)
    assert buckets.shape[-1] == n_hashes * seqlen

    # create sort and unsort
    ticker = jax.lax.tie_in(array, jnp.arange(n_hashes * seqlen))
    buckets_and_t = seqlen * buckets + (ticker % seqlen)
    buckets_and_t = jax.lax.stop_gradient(buckets_and_t)
    sbuckets_and_t, sticker = jax.lax.sort_key_val(
        buckets_and_t, ticker, dimension=-1)
    _, undo_sort = jax.lax.sort_key_val(sticker, ticker, dimension=-1)
    sbuckets_and_t = jax.lax.stop_gradient(sbuckets_and_t)
    sticker = jax.lax.stop_gradient(sticker)
    undo_sort = jax.lax.stop_gradient(undo_sort)

    st = (sticker % seqlen)
    return st, sticker, undo_sort, sbuckets_and_t

  q_st, q_sticker, q_undo_sort, _ = get_sorted_array(query, rng)
  sqk = jnp.take(query, q_st, axis=0)
  sv = jnp.take(value, q_st, axis=0)

  bq = jnp.reshape(sqk, (chunk_size, -1, qdim))
  bk = bq
  bv = jnp.reshape(sv, (chunk_size, -1, vdim))
  if look_both_side:
    # get previous and next chunks
    bk = look_one_back_and_ahead(bk)
    bv = look_one_back_and_ahead(bv)
  else:
    # get previous chunks
    bk = look_one_back(bk)
    bv = look_one_back(bv)

  # compute dot product attention
  dots = jnp.einsum('hie,hje->hij', bq, bk) / (qdim ** 0.5)

  def mask_self_attention(scores, exclude_self=False, masked=False, causal=False):
    # Tie in the key to avoid the mask becoming a constant.
    # This way XLA can construct the mask during computation and fuse it
    # with the attention ops.
    q_info = jax.lax.tie_in(key, jnp.arange(query.shape[0], dtype=jnp.int32))
    q_info = q_info + 1
    q_info = jnp.take(q_info, q_st, axis=0)
    q_info = jnp.reshape(q_info, (chunk_size, -1))
    q_len = q_info.shape[1]

    k_info = jax.lax.tie_in(key, jnp.arange(key.shape[0], dtype=jnp.int32))
    k_info = k_info + 1
    if masked:
      k_info = jnp.where(padding_mask.reshape([-1]), k_info, -k_info)
    k_info = jnp.take(k_info, q_st, axis=0)
    k_info = jnp.reshape(k_info, (chunk_size, -1))
    if look_both_side:
      k_info = look_one_back_and_ahead(k_info)
    else:
      k_info = look_one_back(k_info)
    k_len = k_info.shape[1]

    if exclude_self:
      mask = jax.lax.eq(
          jax.lax.broadcast_in_dim(q_info, shape=(chunk_size, q_len, k_len),
                                   broadcast_dimensions=(0, 1)),
          jax.lax.broadcast_in_dim(k_info, shape=(chunk_size, q_len, k_len),
                                   broadcast_dimensions=(0, 2)))
      mask = mask.astype(jnp.float32)
      scores = scores - 1e5 * mask

    if masked:
      zeros_like_k_info = jnp.zeros_like(k_info)
      mask = jax.lax.lt(k_info, zeros_like_k_info).astype(jnp.float32)
      mask = jax.lax.broadcast_in_dim(mask,
                                      shape=(chunk_size, q_len, k_len),
                                      broadcast_dimensions=(0, 2))
      scores = scores - 1e9 * mask

    if causal:
      mask = jax.lax.lt(
          jax.lax.broadcast_in_dim(q_info, shape=(chunk_size, q_len, k_len),
                                   broadcast_dimensions=(0, 1)),
          jax.lax.broadcast_in_dim(k_info, shape=(chunk_size, q_len, k_len),
                                   broadcast_dimensions=(0, 2)))
      mask = mask.astype(jnp.float32)
      scores = scores - 1e9 * mask

    return scores

  if causal_mask or self_mask or padding_masked:
    dots = mask_self_attention(dots, exclude_self=self_mask, masked=padding_masked, causal=causal_mask)

  dots_logsumexp = logsumexp(dots, axis=-1, keepdims=True)
  slogits = jnp.reshape(dots_logsumexp, [-1])
  dots = jnp.exp(dots - dots_logsumexp)

  x = jnp.matmul(dots, bv)
  x = jnp.reshape(x, [-1, qdim])

  # Unsort
  def get_unsorted_output(x, sticker, undo_sort):
    o = permute_via_gather(x, undo_sort, sticker, axis=0)
    o = jnp.reshape(o, [n_hashes, seqlen, qdim])
    logits = permute_via_sort(slogits, sticker, undo_sort, axis=0)
    logits = jnp.reshape(logits, [total_hashes, seqlen, 1])
    probs = jnp.exp(logits - logsumexp(logits, axis=0, keepdims=True))
    out = jnp.sum(o * probs, axis=0)
    out = jnp.reshape(out, [seqlen, qdim])
    return out

  output = get_unsorted_output(x, q_sticker, q_undo_sort)
  return output



class LSHAttention(nn.Module):
  """Multi-head Reformer Architecture."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            causal_mask=False,
            padding_mask=None,
            key_padding_mask=None,
            cache=None,
            broadcast_dropout=True,
            dropout_rng=None,
            dropout_rate=0.,
            deterministic=False,
            precision=None,
            kernel_init=nn.linear.default_kernel_init,
            bias_init=nn.initializers.zeros,
            bias=True,
            chunk_len=10,
            n_chunks_before=1,
            n_hashes=1,
            n_buckets=10,
            self_mask=False,
            padding_masked=False,
            look_both_side=False,
            qk_length_norm=False,
            output_projection=True):
    """Applies multi-head reformer attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
      inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]`
        or None for self-attention, inn which case key/values will be derived
        from inputs_q.
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      causal_mask: boolean specifying whether to apply a causal mask on the
        attention weights. If True, the output at timestep `t` will not depend
        on inputs at timesteps strictly greater than `t`.
      padding_mask: boolean specifying query tokens that are pad token.
      key_padding_mask: boolean specifying key-value tokens that are pad token.
      cache: an instance of `flax.nn.attention.Cache` used for efficient
        autoregressive decoding.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      bias: bool: whether pointwise QKVO dense transforms use bias.
      chunk_len: int, chunk length.
      n_chunks_before: int, number of chunks before to attend to.
      n_hashes: int, number of hashes.
      n_buckets: int, number of buckets.
      self_mask: bool
      padding_masked: bool
      look_both_side: bool
      qk_length_norm: bool
      output_projection: bool

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    assert causal_mask or not cache, (
        'Caching is only support for causal attention.')

    assert inputs_q.ndim == 3

    if inputs_kv is None:
      inputs_kv = inputs_q

    qkv_features = inputs_q.shape[-1]
    qlength = inputs_q.shape[1]
    orig_seqlen = inputs_q.shape[1]

    # chunk_size = n_hashes * n_buckets

    extra_len = chunk_len - (qlength % chunk_len)
    pad_width = jnp.array([[0, 0], [0, extra_len], [0, 0]])

    inputs_q = jnp.pad(inputs_q, pad_width)
    inputs_kv = jnp.pad(inputs_kv, pad_width)
    if key_padding_mask is not None:
      key_padding_mask = jnp.pad(key_padding_mask, pad_width)
    if padding_mask is not None:
      padding_mask = jnp.pad(padding_mask, pad_width)

    assert qkv_features % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // num_heads

    dense = nn.DenseGeneral.partial(
        axis=-1,
        features=(num_heads, head_dim),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        precision=precision)

    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]
    query, value = (dense(inputs_q, dtype=dtype, name='query'),
                    dense(inputs_kv, dtype=dtype, name='value'))
    key = query

    if key_padding_mask is None:
      key_padding_mask = padding_mask
      if padding_mask is None:
        padding_shape = [key.shape[0], key.shape[1], 1]
        key_padding_mask = jnp.full(padding_shape, True)

    if qk_length_norm:
      query = length_normalized(query)
      key = length_normalized(key)

    attn = jax.vmap(lsh_attention_single_batch,
                    in_axes=(0, 0, 0, 0,
                             None, None, None, None, None, None))

    logging.info(query)
    out = attn(query, key, value, key_padding_mask,
               n_buckets, n_hashes, self_mask, padding_masked, look_both_side, causal_mask)
    logging.info(out)

    if output_projection:
      out = nn.DenseGeneral(out,
                            features=qkv_features,
                            axis=(-2, -1),
                            kernel_init=kernel_init,
                            bias_init=bias_init,
                            bias=bias,
                            dtype=dtype,
                            precision=precision,
                            name='out')
    out = out[:, :orig_seqlen, :]
    return out


LSHSelfAttention = LSHAttention.partial(inputs_kv=None)
