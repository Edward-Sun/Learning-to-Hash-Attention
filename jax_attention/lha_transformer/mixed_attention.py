# Lint as: python3
"""Attention modules for mixing local attention and LHA attention."""

import jax
import jax.numpy as jnp
from absl import logging
from flax import nn
from jax import random
from jax_attention.lha_transformer.local_attention import local_attention
from jax_attention.lha_transformer.lha_attention import lha_attention


class LHATransformerAttention(nn.Module):
  """Multi-head LHA Transformer Architecture."""

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
            n_buckets=8,
            output_projection=True,
            local_num_heads=0,
            local_block_size=0,
            relative_pos_emb=False,
            abs_pos_emb_per_layer=False,
            pos_emb=None,
            r_w_bias=None,
            r_r_bias=None,
            share_qk_bucket=False,
            prf_temperature=1.0,
            cluster_score=0.1,
            nb_features=256,
            lha_score_func="linear"):
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
      n_buckets: int, number of buckets.
      output_projection: bool
      local_num_heads: int
      local_block_size: int
      relative_pos_emb: bool
      abs_pos_emb_per_layer: bool
      pos_emb: tensor
      r_w_bias: tensor
      r_r_bias: tensor
      share_qk_bucket: bool
      prf_temperature: float
      cluster_score: float
      nb_features: int
      lha_score_func: str

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
    batch_size = inputs_q.shape[0]

    chunk_len = ((qlength - 1) // n_buckets) + 1
    if local_block_size == 0:
      local_block_size = chunk_len

    extra_len = local_block_size - (((qlength - 1) % local_block_size) + 1)
    extra_len = max(extra_len, n_buckets * local_block_size - orig_seqlen)
    pad_width = jnp.array([[0, 0], [0, extra_len], [0, 0]])

    inputs_q = jnp.pad(inputs_q, pad_width)
    inputs_kv = jnp.pad(inputs_kv, pad_width)

    qlength = inputs_q.shape[1]

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
        dtype=dtype,
        precision=precision)

    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]

    query, key, value = (dense(inputs_q, name='query'),
                         dense(inputs_kv, name='key'),
                         dense(inputs_kv, name='value'))

    if pos_emb is not None:
      pos_emb = pos_emb[-inputs_q.shape[1]:]
    pe_dense = nn.DenseGeneral.partial(
        axis=-1,
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        precision=precision,
        dtype=dtype)

    if key_padding_mask is None:
      key_padding_mask = padding_mask
      if padding_mask is None:
        padding_shape = [inputs_kv.shape[0], inputs_kv.shape[1], 1]
        key_padding_mask = jnp.full(padding_shape, True)

    def get_cluster_scores(query, key, num_heads):
      if lha_score_func == "linear":
        shared_kernel = self.param('score_kernel',
                                          (num_heads, head_dim, n_buckets),
                                          initializer=kernel_init)
        query_kernel = self.param('query_score_kernel',
                                  (num_heads, head_dim, n_buckets),
                                  initializer=kernel_init)
        key_kernel = self.param('key_score_kernel',
                                (num_heads, head_dim, n_buckets),
                                initializer=kernel_init)

        q_cluster_score = jnp.einsum(
            'blhd,hdk->blhk',
            query,
            (1-cluster_score) * shared_kernel + cluster_score * query_kernel,
        )
        k_cluster_score = jnp.einsum(
            'blhd,hdk->blhk',
            key,
            (1-cluster_score) * shared_kernel + cluster_score * key_kernel,
        )
      else:
        raise ValueError("Unknown lha_score_func: %s" % lha_score_func)

      if share_qk_bucket:
        q_cluster_score = (q_cluster_score + k_cluster_score) / 2
        k_cluster_score = q_cluster_score

      return q_cluster_score, k_cluster_score

    extra_loss = None

    if local_num_heads == 0:
      if abs_pos_emb_per_layer:
        query_pe = pe_dense(pos_emb, features=(num_heads, head_dim),
                            name='pe_query')
        key_pe = pe_dense(pos_emb, features=(num_heads, head_dim),
                          name='pe_key')
        query += query_pe
        key += key_pe
      query_cluster_score, key_cluster_score = get_cluster_scores(jax.lax.stop_gradient(query),
                                                                  jax.lax.stop_gradient(key),
                                                                  num_heads)

      out = lha_attention(query, key, value, key_padding_mask,
                          query_cluster_score, key_cluster_score,
                          n_buckets, causal_mask, jnp.float32, dropout_rate,
                          deterministic, nb_features,
                          attention_temperature=prf_temperature,
                          share_qk_bucket=share_qk_bucket)
      out, extra_loss = out
    elif local_num_heads == num_heads:
      r_key = None
      if relative_pos_emb:
        r_key = pe_dense(pos_emb, features=(local_num_heads, head_dim),
                         name='r_key')

      out = local_attention(query, key, value, key_padding_mask, local_block_size,
                            causal_mask, jnp.float32, deterministic, dropout_rate,
                            relative_pos_emb, r_key, r_w_bias, r_r_bias)
    else:
      query_l, query_g = jnp.split(query, (local_num_heads,), axis=-2)
      key_l, key_g = jnp.split(key, (local_num_heads,), axis=-2)
      value_l, value_g = jnp.split(value, (local_num_heads,), axis=-2)

      r_key = None
      pos_bias = None
      if relative_pos_emb:
        r_w_bias = r_w_bias[:local_num_heads]
        r_r_bias = r_w_bias[:local_num_heads]
        r_key = pe_dense(pos_emb, features=(local_num_heads, head_dim),
                         name='r_key')
      elif abs_pos_emb_per_layer:
        query_pe_l = pe_dense(pos_emb, features=(local_num_heads, head_dim),
                              name='pe_query_local')
        key_pe_l = pe_dense(pos_emb, features=(local_num_heads, head_dim),
                            name='pe_key_local')
        query_l += query_pe_l
        key_l += key_pe_l

      out_l = local_attention(query_l, key_l, value_l, key_padding_mask, local_block_size,
                              causal_mask, jnp.float32, deterministic, dropout_rate,
                              relative_pos_emb, r_key, r_w_bias, r_r_bias)

      if abs_pos_emb_per_layer:
        query_pe = pe_dense(pos_emb, features=(num_heads - local_num_heads, head_dim),
                            name='pe_query')
        key_pe = pe_dense(pos_emb, features=(num_heads - local_num_heads, head_dim),
                          name='pe_key')
        query_g += query_pe
        key_g += key_pe
      query_cluster_score, key_cluster_score = get_cluster_scores(jax.lax.stop_gradient(query_g),
                                                                  jax.lax.stop_gradient(key_g),
                                                                  num_heads - local_num_heads)
      out_g = lha_attention(query_g, key_g, value_g, key_padding_mask,
                            query_cluster_score, key_cluster_score,
                            n_buckets, causal_mask, jnp.float32, dropout_rate,
                            deterministic, nb_features,
                            attention_temperature=prf_temperature,
                            share_qk_bucket=share_qk_bucket)
      out_g, extra_loss = out_g
      out = jnp.concatenate([out_l, out_g], axis=-2)

    if output_projection:
      out = nn.DenseGeneral(out,
                            axis=(-2, -1),
                            features=qkv_features,
                            kernel_init=kernel_init,
                            bias_init=bias_init,
                            bias=bias,
                            dtype=dtype,
                            precision=precision,
                            name='out')
    out = out[:, :orig_seqlen]
    return out, extra_loss


LHATransformerSelfAttention = LHATransformerAttention.partial(inputs_kv=None)
