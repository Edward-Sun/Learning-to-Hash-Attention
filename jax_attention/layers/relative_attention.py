# Lint as: python3
"""Common layers used in models."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nn
from flax.nn.stochastic import make_rng
from jax import lax

from common_layers import rel_shift


class RelativeMultiHeadDotProductAttention(nn.Module):
  """The `flax.nn` module is Deprecated, use `flax.linen` instead.
  Learn more and find an upgrade guide at
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Multi-head dot-product attention with abs/rel positional encoding."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            attention_axis=None,
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
            pos_emb=None,
            r_w_bias=None,
            r_r_bias=None,
            abs_pos_emb=False):
    """Applies multi-head dot product attention on the input data.

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
      attention_axis: axes over which the attention is applied ( 'None' means
        attention over all axes, but batch, heads, and features).
      causal_mask: boolean specifying whether to apply a causal mask on the
        attention weights. If True, the output at timestep `t` will not depend
        on inputs at timesteps strictly greater than `t`.
      padding_mask: boolean specifying query tokens that are pad token w/ False.
      key_padding_mask: boolean specifying key-value tokens that are pad token
        w/ False.
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
      attention_fn: dot_product_attention or compatible function. Accepts
      query, key, value, and returns output of shape
      `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      pos_emb: tensor
      r_w_bias: tensor
      r_r_bias: tensor
      abs_pos_emb: bool

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    assert causal_mask or not cache, (
        'Caching is only support for causal attention.')

    if inputs_kv is None:
      inputs_kv = inputs_q

    is_self_attention = inputs_kv is inputs_q

    if attention_axis is None:
      attention_axis = tuple(range(1, inputs_q.ndim - 1))

    features = out_features or inputs_q.shape[-1]
    qkv_features = qkv_features or inputs_q.shape[-1]

    assert qkv_features % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // num_heads

    dense = nn.linear.DenseGeneral.partial(
        axis=-1,
        features=(num_heads, head_dim),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        precision=precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]

    if abs_pos_emb:
      pe_dense = nn.linear.DenseGeneral.partial(
          axis=-1,
          features=qkv_features,
          kernel_init=kernel_init,
          bias_init=bias_init,
          bias=bias,
          precision=precision,
          dtype=dtype)

      query_pe = pe_dense(pos_emb[-inputs_q.shape[1]:])
      key_pe = pe_dense(pos_emb[-inputs_kv.shape[1]:])

      query, key, value = (dense(inputs_q + query_pe, dtype=dtype, name='query'),
                           dense(inputs_kv + key_pe, dtype=dtype, name='key'),
                           dense(inputs_kv, dtype=dtype, name='value'))
    else:
      query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                           dense(inputs_kv, dtype=dtype, name='key'),
                           dense(inputs_kv, dtype=dtype, name='value'))

    if cache:
      assert isinstance(cache,
                        nn.attention.Cache), 'cache must be an instance of Cache'
      if self.is_initializing():
        cache.store(np.array((key.ndim,) + key.shape[-2:], dtype=np.int32))
      else:
        cache_entry = cache.retrieve(None)
        expected_shape = list(cache_entry.key.shape[:-2])
        for attn_dim in attention_axis:
          expected_shape[attn_dim] = 1
        expected_shape = tuple(expected_shape) + inputs_q.shape[-1:]
        if expected_shape != inputs_q.shape:
          raise ValueError('Invalid shape provided, '
                           'expected shape %s instead got %s.' %
                           (expected_shape, inputs_q.shape))

        if not isinstance(cache_entry, nn.attention._CacheEntry):
          raise ValueError('Cache is not initialized.')

        cshape = cache_entry.key.shape
        indices = [0] * len(cshape)
        i = cache_entry.i
        attn_size = np.prod(np.take(cshape, attention_axis))
        for attn_dim in attention_axis:
          attn_size //= cshape[attn_dim]
          indices[attn_dim] = i // attn_size
          i = i % attn_size

        key = lax.dynamic_update_slice(cache_entry.key, key, indices)
        value = lax.dynamic_update_slice(cache_entry.value, value, indices)
        one = jnp.array(1, jnp.uint32)
        cache_entry = cache_entry.replace(i=cache_entry.i + one,
                                          key=key,
                                          value=value)
        cache.store(cache_entry)

    # create attention masks
    mask_components = []

    if causal_mask:
      if cache and not self.is_initializing():
        bias_pre_shape = (1,) * (key.ndim - 1)
        attn_shape = tuple(np.take(key.shape, attention_axis))
        attn_size = np.prod(attn_shape)
        ii = jnp.arange(attn_size, dtype=jnp.uint32)
        mask = ii < cache_entry.i
        mask_components.append(mask.reshape(bias_pre_shape + attn_shape))
      else:
        mask_components.append(
          nn.attention._make_causal_mask(key, attention_axis))

    if (padding_mask is not None or key_padding_mask is not None) and not cache:
      if key_padding_mask is None:
        if is_self_attention:
          key_padding_mask = padding_mask
        else:
          key_padding_shape = [inputs_kv.shape[dim] for dim in attention_axis]
          key_padding_mask = jnp.full(key_padding_shape, True)
      if padding_mask is None:
        if is_self_attention:
          padding_mask = key_padding_mask
        else:
          padding_shape = [inputs_q.shape[dim] for dim in attention_axis]
          padding_mask = jnp.full(padding_shape, True)

      padding_mask = nn.attention.make_padding_mask(
          padding_mask_query=padding_mask,
          padding_mask_key=key_padding_mask,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis)
      mask_components.append(padding_mask)

    if mask_components:
      attention_mask = mask_components[0]
      for component in mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)

      # attention mask in the form of attention bias
      attention_bias = lax.select(
          attention_mask > 0, jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    depth = query.shape[-1]
    if abs_pos_emb:
      if len(key.shape) == 4:
        query = query / jnp.sqrt(depth).astype(dtype)
        attn_score = jnp.einsum('bind,bjnd->bnij', query, key)
      else:
        raise NotImplementedError

    else:
      # apply attention
      rw_query = query + r_w_bias
      rr_query = query + r_r_bias

      pos_emb = pos_emb[-key.shape[1]:]

      r_key = dense(pos_emb, dtype=dtype, name='r_key')

      if len(key.shape) == 4:
        AC = jnp.einsum('bind,bjnd->ijbn', rw_query, key)
        BD = jnp.einsum('bind,jnd->ijbn', rr_query, r_key)
        BD = rel_shift(BD)
        attn_score = (AC + BD) / jnp.sqrt(depth).astype(dtype)
        attn_score = attn_score.transpose((2, 3, 0, 1))
      else:
        raise NotImplementedError

    if attention_bias is not None:
      attn_score = attn_score + attention_bias

    attn_score = nn.activation.softmax(attn_score, axis=-1)
    attn_score = attn_score.astype(dtype)

    axis = (1, )
    batch_dims = (0, 2)
    batch_dims_t = tuple(range(len(batch_dims)))
    norm_dims = tuple(range(attn_score.ndim - len(axis), attn_score.ndim))

    # apply dropout
    if not deterministic and dropout_rate > 0.:
      if dropout_rng is None:
        dropout_rng = make_rng()
      keep_prob = jax.lax.tie_in(attn_score, 1.0 - dropout_rate)
      if broadcast_dropout:
        # dropout is broadcast across the batch+head+non-attention dimension
        dropout_dims = attn_score.shape[-(2 * len(axis)):]
        dropout_shape = (tuple([1] * len(batch_dims_t)) + dropout_dims)
        keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
      else:
        keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_score.shape)
      multiplier = (keep.astype(attn_score.dtype) /
                    jnp.asarray(keep_prob, dtype=dtype))
      attn_score = attn_score * multiplier

    # compute the new values given the attention weights
    qk_perm = batch_dims + axis + (key.ndim - 1,)
    v_perm = batch_dims + (value.ndim - 1,) + axis
    value = value.transpose(v_perm)

    wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
    y = lax.dot_general(
        attn_score,
        value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
        precision=precision)

    perm_inv = nn.attention._invert_perm(qk_perm)
    y = y.transpose(perm_inv)

    # back to the original inputs dimensions
    out = nn.linear.DenseGeneral(
        y,
        features=features,
        axis=(-2, -1),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        dtype=dtype,
        precision=precision,
        name='out')

    return out


RelativeSelfAttention = RelativeMultiHeadDotProductAttention.partial(inputs_kv=None)
