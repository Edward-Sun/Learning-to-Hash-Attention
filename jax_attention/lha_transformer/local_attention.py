# Lint as: python3
"""Attention modules for local Transformer model."""

import jax
import jax.numpy as jnp
from flax import nn


def local_attention(query, key, value, padding_mask, block_size,
                    causal_mask, dtype, deterministic, dropout_rate,
                    relative_pos_emb, r_key, r_w_bias, r_r_bias):
  # break_into_blocks
  batch_size, qlength, num_heads, qdim = query.shape
  klength = key.shape[1]
  vdim = value.shape[-1]
  n_blocks = qlength // block_size
  query = query.reshape(batch_size, n_blocks, block_size, num_heads, qdim)
  query *= (qdim ** -0.5)

  # break_into_memory_blocks
  def break_into_memory_blocks(x, masked=False):
    dim_1 = x.shape[0]
    dim_m2 = x.shape[-2]
    dim_m1 = x.shape[-1]
    x = x.reshape(dim_1, n_blocks, block_size, dim_m2, dim_m1)
    if masked:
      xlb1 = jnp.concatenate([x[:, -1:], x[:, :-1]], axis=1)
      xlb2 = jnp.concatenate([x[:, -2:], x[:, :-2]], axis=1)
      return jnp.concatenate([xlb2, xlb1, x], axis=2)
    else:
      xlb = jnp.concatenate([x[:, -1:], x[:, :-1]], axis=1)
      xla = jnp.concatenate([x[:, 1:], x[:, :1]], axis=1)
      return jnp.concatenate([x, xlb, xla], axis=2)

  key = break_into_memory_blocks(key, causal_mask)
  value = break_into_memory_blocks(value, causal_mask)

  def rel_shift(x):
    x_size = x.shape
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[1] = (1, 0)  # Padding on axis=1
    x = jnp.pad(x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
    x = jnp.reshape(x, (x.shape[1], x.shape[0]) + x.shape[2:])
    x = x[1:]
    x = jnp.reshape(x, x_size)
    return x

  if relative_pos_emb:
    if r_key is None:
      attention_raw_score = jnp.einsum('bknhd,bkmhd->bknhm', query, key)
    else:
      rw_query = query + r_w_bias
      rr_query = query + r_r_bias
      r_key = r_key[-key.shape[2]:]
      AC = jnp.einsum('bknhd,bkmhd->nmbkh', rw_query, key)
      BD = jnp.einsum('bknhd,mhd->nmbkh', rr_query, r_key)
      BD = rel_shift(BD)
      attention_raw_score = (AC + BD) / jnp.sqrt(vdim).astype(dtype)
      attention_raw_score = attention_raw_score.transpose((2, 3, 0, 4, 1))
  else:
    attention_raw_score = jnp.einsum('bknhd,bkmhd->bknhm', query, key)

  if padding_mask is not None:
    padding_mask_float = jax.lax.select(
        padding_mask > 0, jnp.full(padding_mask.shape, 0.).astype(dtype),
        jnp.full(padding_mask.shape, -1e10).astype(dtype))
    padding_mask_float = padding_mask_float.reshape(batch_size, -1, 1, 1)
    padding_mask_float = break_into_memory_blocks(
        padding_mask_float, causal_mask)
    padding_mask_float = padding_mask_float.reshape(batch_size, n_blocks, 1, 1, -1)
    attention_raw_score += padding_mask_float

  if causal_mask:
    q_info = jax.lax.tie_in(query, jnp.arange(qlength, dtype=jnp.int32))
    q_info = q_info.reshape(1, n_blocks, block_size, 1, 1)

    k_info = jax.lax.tie_in(key, jnp.arange(klength, dtype=jnp.int32))
    k_info = k_info.reshape(1, -1, 1, 1)
    k_info = break_into_memory_blocks(k_info, causal_mask)
    k_info = k_info.reshape(1, n_blocks, 1, 1, -1)

    causal_mask_float = (k_info <= q_info)
    causal_mask_float = jax.lax.select(
        causal_mask_float,
        jnp.full(causal_mask_float.shape, 0.).astype(dtype),
        jnp.full(causal_mask_float.shape, -1e10).astype(dtype))
    attention_raw_score += causal_mask_float
  attention_score = jax.nn.softmax(attention_raw_score, axis=-1)

  attention_score = nn.dropout(
      attention_score,
      rate=dropout_rate,
      deterministic=deterministic)

  new_value = jnp.einsum('bknhm,bkmhd->bknhd', attention_score, value)
  new_value = new_value.reshape(batch_size, qlength, num_heads, vdim)
  return new_value
