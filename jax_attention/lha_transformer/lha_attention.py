# Lint as: python3
"""Attention modules for LHA-based sparse attention Transformer model."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import nn

import functools
from jax_attention.performer.fast_attention import \
  GaussianOrthogonalRandomMatrix, nonnegative_softmax_kernel_feature_creator


def length_normalized(x, epsilon=1e-6):
  norm_inputs = nn.LayerNorm(x, epsilon=epsilon, bias=False, scale=False)
  return norm_inputs


def lha_attention(query, key, value, padding_mask,
                  query_cluster_score, key_cluster_score,
                  n_buckets, causal_mask, dtype, dropout_rate,
                  deterministic, nb_features, attention_temperature=1.0, share_qk_bucket=True, normalize_qk=True):
  """LHA Attention layer.

  Args:
    query: [batch_size, qlength, num_heads, qdim] tensor.
    key: [batch_size, klength, num_heads, qdim] tensor.
    value: [batch_size, klength, num_heads, vdim] tensor.
    padding_mask: [batch_size, qlength] tensor.
    query_cluster_score: [batch_size, num_heads, qlength, n_buckets] tensor.
    key_cluster_score: [batch_size, num_heads, klength, n_buckets] tensor.
    n_buckets: number of buckets.
    causal_mask: bool, whether to apply causal mask.
    dtype: dtype of the computation (default: float32).
    dropout_rate: dropout rate (default: 0.0).
    deterministic: bool, deterministic or not (to apply dropout).
    nb_features: number of features for the random feature map.
    attention_temperature: temperature for the attention softmax.
    share_qk_bucket: bool, whether to share query and key buckets.
    normalize_qk: bool, whether to normalize query and key.

  Returns:
    new_value: [batch_size, qlength, num_heads, vdim] tensor.
    extra_loss: float, extra loss for the LHA attention training.
  """
  batch_size = query.shape[0]
  qlength = query.shape[1]
  klength = key.shape[1]
  assert qlength == klength
  num_heads = query.shape[2]
  qdim, vdim = query.shape[-1], value.shape[-1]

  if causal_mask and not share_qk_bucket:
    raise NotImplementedError

  chunk_len = qlength // n_buckets
  chunk_len = int(chunk_len * (2 ** 0.5))

  query = jnp.transpose(query, (0, 2, 1, 3)).reshape(batch_size * num_heads, qlength, qdim)
  key = jnp.transpose(key, (0, 2, 1, 3)).reshape(batch_size * num_heads, klength, qdim)
  value = jnp.transpose(value, (0, 2, 1, 3)).reshape(batch_size * num_heads, klength, vdim)

  if normalize_qk:
    query = length_normalized(query)
    key = length_normalized(key)

  query_prod = jnp.transpose(query_cluster_score, (0, 2, 3, 1)).reshape(batch_size * num_heads, n_buckets, qlength)
  key_prod = jnp.transpose(key_cluster_score, (0, 2, 3, 1)).reshape(batch_size * num_heads, n_buckets, klength)

  masked_query_prod = query_prod
  masked_query_prod = jax.nn.softmax(masked_query_prod, axis=-2)

  masked_key_prod = key_prod
  masked_key_prod = jax.nn.softmax(masked_key_prod, axis=-2)

  padding_mask_float = None
  if padding_mask is not None:
    padding_mask_float = jax.lax.select(
        padding_mask > 0, jnp.full(padding_mask.shape, 0.).astype(dtype),
        jnp.full(padding_mask.shape, -1e10).astype(dtype))
    padding_mask_float = jnp.reshape(padding_mask_float, (batch_size, 1, -1))
    padding_mask_float = jnp.tile(padding_mask_float, (1, num_heads, 1))
    padding_mask_float = jnp.reshape(padding_mask_float, (batch_size * num_heads, 1, -1))
    masked_query_prod = masked_query_prod + padding_mask_float
    masked_key_prod = masked_key_prod + padding_mask_float

  query_top_k_prod, query_top_k_idx = jax.lax.top_k(masked_query_prod, chunk_len)
  key_top_k_prod, key_top_k_idx = jax.lax.top_k(masked_key_prod, chunk_len)

  query_top_k = jnp.take_along_axis(query, query_top_k_idx.reshape((batch_size * num_heads, n_buckets * chunk_len, 1)),
                                    axis=1).reshape((batch_size * num_heads, n_buckets, chunk_len, qdim))
  key_top_k = jnp.take_along_axis(key, key_top_k_idx.reshape((batch_size * num_heads, n_buckets * chunk_len, 1)),
                                  axis=1).reshape((batch_size * num_heads, n_buckets, chunk_len, qdim))
  value_top_k = jnp.take_along_axis(value, key_top_k_idx.reshape((batch_size * num_heads, n_buckets * chunk_len, 1)),
                                    axis=1).reshape((batch_size * num_heads, n_buckets, chunk_len, vdim))

  query_top_k = query_top_k * (qdim ** -0.5)
  raw_attention_score = jnp.einsum('bkld,bkmd->bklm', query_top_k, key_top_k)

  if padding_mask is not None:
    padding_mask_top_k = jnp.take_along_axis(
        padding_mask_float.reshape(batch_size * num_heads, klength),
        key_top_k_idx.reshape((batch_size * num_heads, n_buckets * chunk_len)),
        axis=1).reshape((batch_size * num_heads, n_buckets, 1, chunk_len))
    raw_attention_score += padding_mask_top_k

  if causal_mask:
    q_info = jax.lax.tie_in(query, jnp.arange(query.shape[1], dtype=jnp.int32))
    q_info = q_info.reshape(1, -1)
    q_info = jnp.tile(q_info, (batch_size * num_heads, 1))
    q_info_top_k = jnp.take_along_axis(
        q_info, query_top_k_idx.reshape((batch_size * num_heads, n_buckets * chunk_len)), axis=1)
    q_info_top_k = q_info_top_k.reshape(batch_size * num_heads, n_buckets, chunk_len, 1)

    k_info = jax.lax.tie_in(key, jnp.arange(key.shape[1], dtype=jnp.int32))
    k_info = k_info.reshape(1, -1)
    k_info = jnp.tile(k_info, (batch_size * num_heads, 1))
    k_info_top_k = jnp.take_along_axis(
        k_info, key_top_k_idx.reshape((batch_size * num_heads, n_buckets * chunk_len)), axis=1)
    k_info_top_k = k_info_top_k.reshape(batch_size * num_heads, n_buckets, 1, chunk_len)

    causal_mask_float = (k_info_top_k <= q_info_top_k)

    causal_mask_float = jax.lax.select(
        causal_mask_float,
        jnp.full(causal_mask_float.shape, 0.).astype(dtype),
        jnp.full(causal_mask_float.shape, -1e10).astype(dtype))

    raw_attention_score += causal_mask_float

  sum_attention_score = logsumexp(raw_attention_score, axis=-1)
  attention_prob = jax.nn.softmax(raw_attention_score, axis=-1)
  attention_prob = nn.dropout(
      attention_prob,
      rate=dropout_rate,
      deterministic=deterministic)
  result_top_k = jnp.einsum('bklm,bkmd->bkld', attention_prob, value_top_k)

  def merge_attention(_sum_attention_score, _result_top_k, _query_top_k_idx):
    _sum_attention_score = _sum_attention_score.reshape(-1)
    _result_top_k = _result_top_k.reshape(-1, vdim)
    _query_top_k_idx = _query_top_k_idx.reshape(-1)

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    max_weight = jnp.zeros(qlength).astype(dtype)
    max_weight = jax.lax.scatter_max(
        operand=max_weight,
        scatter_indices=_query_top_k_idx.reshape(-1, 1),
        updates=_sum_attention_score,
        dimension_numbers=dnums,
        indices_are_sorted=False,
        unique_indices=False,
    )
    _sum_attention_score = jnp.exp(_sum_attention_score - jax.lax.stop_gradient(max_weight[_query_top_k_idx]))

    weight = jnp.zeros(qlength).astype(dtype)
    weight = jax.lax.scatter_add(
        operand=weight,
        scatter_indices=_query_top_k_idx.reshape(-1, 1),
        updates=_sum_attention_score,
        dimension_numbers=dnums,
        indices_are_sorted=False,
        unique_indices=False,
    )
    weighted_result = _result_top_k * (_sum_attention_score / (weight[_query_top_k_idx])).reshape(-1, 1)

    dnums2 = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    _merged_result = jnp.zeros((qlength, vdim)).astype(dtype)
    _merged_result = jax.lax.scatter_add(
        operand=_merged_result,
        scatter_indices=_query_top_k_idx.reshape(-1, 1),
        updates=weighted_result,
        dimension_numbers=dnums2,
        indices_are_sorted=False,
        unique_indices=False,
    )
    return _merged_result

  merge_attention = jax.vmap(merge_attention, in_axes=(0, 0, 0))
  merged_result = merge_attention(sum_attention_score, result_top_k, query_top_k_idx)
  merged_result = merged_result.reshape((batch_size, num_heads, qlength, vdim)).transpose((0, 2, 1, 3))

  if deterministic:
    return merged_result, 0.0

  def get_performer_feature(query, key, full_query, full_key, precision=None, numerical_stabilizer=1e-6):
    query_seed = jax.lax.convert_element_type(
        jnp.ceil(jnp.sum(query) * 10000000.0), jnp.int32)
    rng = jax.random.PRNGKey(query_seed)
    matrixrng, _ = jax.random.split(rng)
    matrix_creator = functools.partial(
        GaussianOrthogonalRandomMatrix,
        nb_features, qdim, scaling=0)
    projection_matrix = matrix_creator(key=matrixrng).get_2d_array()

    def kernel_feature_creator(data,
                               projection_matrix,
                               attention_dims_t,
                               batch_dims_t,
                               precision,
                               is_query,
                               normalize_data=True):
      return nonnegative_softmax_kernel_feature_creator(
          data, projection_matrix, attention_dims_t, batch_dims_t, precision,
          is_query, normalize_data, numerical_stabilizer)

    batch_dims_t = (0, 1)
    attention_dims_t = (1, 2)
    query_prime = kernel_feature_creator(query, projection_matrix,
                                         attention_dims_t, batch_dims_t,
                                         precision, False)
    key_prime = kernel_feature_creator(key, projection_matrix,
                                       attention_dims_t, batch_dims_t,
                                       precision, True)

    batch_dims_t = (0,)
    attention_dims_t = (1,)
    full_query_prime = kernel_feature_creator(full_query, projection_matrix,
                                              attention_dims_t, batch_dims_t,
                                              precision, False)
    full_key_prime = kernel_feature_creator(full_key, projection_matrix,
                                            attention_dims_t, batch_dims_t,
                                            precision, True)

    return query_prime, key_prime, full_query_prime, full_key_prime

  real_query = attention_temperature * jax.lax.stop_gradient(query_top_k)
  real_key = attention_temperature * jax.lax.stop_gradient(key_top_k)
  real_full_query = attention_temperature * jax.lax.stop_gradient(query)
  real_full_key = attention_temperature * jax.lax.stop_gradient(key)
  query_prime, key_prime, full_query_prime, full_key_prime = get_performer_feature(
      real_query, real_key, real_full_query, real_full_key)

  pf_target_query_cluster_score = jnp.einsum("blh,bkh->blk", full_query_prime, key_prime.sum(axis=2))
  pf_target_query_cluster_score = (
      pf_target_query_cluster_score / pf_target_query_cluster_score.sum(axis=-1, keepdims=True))

  pf_sum_atttention_per_query = jnp.einsum("bklh,bh->bkl", query_prime, full_key_prime.sum(axis=1))
  query_prime = query_prime / jnp.expand_dims(pf_sum_atttention_per_query, axis=-1)
  pf_target_key_cluster_score = jnp.einsum("bkh,bmh->bmk", query_prime.sum(axis=2), full_key_prime)
  pf_target_key_cluster_score = (
      pf_target_key_cluster_score / pf_target_key_cluster_score.sum(axis=-1, keepdims=True))

  if causal_mask and not share_qk_bucket:
    pf_target_query_cluster_score = pf_target_query_cluster_score[:, 1:]
    pf_target_key_cluster_score = pf_target_key_cluster_score[:, 1:]

  query_cluster_score = query_cluster_score.transpose((0, 2, 1, 3)).reshape(batch_size * num_heads, -1, n_buckets)
  key_cluster_score = key_cluster_score.transpose((0, 2, 1, 3)).reshape(batch_size * num_heads, -1, n_buckets)

  extra_loss = (pf_target_query_cluster_score * nn.log_softmax(query_cluster_score)
                - pf_target_query_cluster_score * jnp.log(pf_target_query_cluster_score))
  extra_loss = (extra_loss + pf_target_key_cluster_score * nn.log_softmax(key_cluster_score)
                - pf_target_key_cluster_score * jnp.log(pf_target_key_cluster_score))
  extra_loss = - extra_loss.sum(axis=-1).mean()
  return merged_result, extra_loss
