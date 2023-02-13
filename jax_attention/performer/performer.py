# Lint as: python3
"""Performer-based language models."""
import functools

from absl import logging
from flax import nn
import jax.numpy as jnp
from jax_attention.layers import common_layers
import jax_attention.performer.fast_attention as fast_attention


# TODO(phillypham): Either branch this implementation or set up CopyBara.

_ATTENTION_FNS = {
    'dot_product':
        lambda qkv_dim, unidirectional=False: nn.dot_product_attention,
    'softmax':
        fast_attention.make_fast_softmax_attention,
    'generalized':
        fast_attention.make_fast_generalized_attention,
}
_DEFAULT_ATTENTION_FN_CLS = 'generalized'


def _make_attention_fn(attention_fn_cls, attention_fn_kwargs=None):
  attention_fn = (
      _ATTENTION_FNS[attention_fn_cls]
      if isinstance(attention_fn_cls, str) else attention_fn_cls)
  return (attention_fn if attention_fn_kwargs is None else functools.partial(
      attention_fn, **attention_fn_kwargs))


class PerformerBlock(nn.Module):
  """Performer layer (https://arxiv.org/abs/2006.03555)."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            cache=None,
            attention_fn_cls=_DEFAULT_ATTENTION_FN_CLS,
            attention_fn_kwargs=None):
    """Applies PerformerBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32).
      inputs_segmentation: input segmentation info for packed examples.
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      cache: flax autoregressive cache for fast decoding.
      attention_fn_cls: Attention function key or callable.
      attention_fn_kwargs: Keywords to pass to `attention_fn_cls`.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    attention_fn = _make_attention_fn(attention_fn_cls, attention_fn_kwargs)(
        qkv_dim // num_heads, unidirectional=causal_mask)
    x = nn.LayerNorm(inputs)
    x = nn.SelfAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=causal_mask,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache,
        attention_fn=attention_fn)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(x)
    y = common_layers.MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


class PerformerEncoder(nn.Module):
  """Performer Model Encoder."""

  def apply(self,
            inputs,
            vocab_size,
            inputs_positions=None,
            inputs_segmentation=None,
            shared_embedding=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            dtype=jnp.float32,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,
            train=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            learn_pos_emb=False,
            classifier=False,
            classifier_pool='CLS',
            num_classes=10,
            attention_fn_cls=_DEFAULT_ATTENTION_FN_CLS,
            attention_fn_kwargs=None):
    """Applies Performer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      learn_pos_emb: boolean, if learn the positional embedding or use the
        sinusoidal positional embedding.
      classifier: boolean, for classification mode (output N-class logits)
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.
      attention_fn_cls: Attention function key or callable.
      attention_fn_kwargs: Keywords to pass to `attention_fn_cls`.

    Returns:
      output of a transformer encoder or logits if classifier_mode is true.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[..., None]

    # Input Embedding
    if shared_embedding is None:
      input_embed = nn.Embed.partial(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)

    if classifier and classifier_pool == 'CLS':
      cls = self.param('cls', (1, 1, emb_dim), nn.initializers.zeros)
      cls = jnp.tile(cls, [x.shape[0], 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
      max_len += 1
      logging.info(src_padding_mask.shape)
      src_padding_mask = jnp.concatenate(
          [src_padding_mask[:, :1], src_padding_mask], axis=1)

    pe_init = nn.initializers.normal(stddev=0.02) if learn_pos_emb else None
    x = common_layers.AddPositionEmbs(
        x,
        inputs_positions=inputs_positions,
        posemb_init=pe_init,
        max_len=max_len,
        name='posembed_input')
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    if use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input
    for lyr in range(num_layers):
      x = PerformerBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          attention_fn_cls=attention_fn_cls,
          attention_fn_kwargs=attention_fn_kwargs,
          name=f'encoderblock_{lyr}')
    encoded = nn.LayerNorm(x, dtype=dtype, name='encoder_norm')

    if classifier:
      encoded = common_layers.classifier_head(
          encoded, num_classes, mlp_dim, pooling_mode=classifier_pool)
    return encoded


class PerformerDecoder(nn.Module):
  """Performer Decoder."""

  def apply(self,
            inputs,
            vocab_size,
            emb_dim=512,
            num_heads=8,
            dtype=jnp.float32,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            shift=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            cache=None,
            attention_fn_cls=_DEFAULT_ATTENTION_FN_CLS,
            attention_fn_kwargs=None):
    """Applies Performer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      emb_dim: dimension of embedding
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: bool: if model is training.
      shift: bool: if we right-shift input - this is only disabled for fast,
        looped single-token autoregressive decoding.
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      cache: flax autoregressive cache for fast decoding.
      attention_fn_cls: Attention function key or callable.
      attention_fn_kwargs: Keywords to pass to `attention_fn_cls`.

    Returns:
      output of a transformer decoder.
    """
    padding_mask = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)[..., None]
    assert inputs.ndim == 2  # (batch, len)
    x = inputs
    if shift:
      x = common_layers.shift_right(x)
    x = x.astype('int32')
    x = common_layers.Embed(
        x, num_embeddings=vocab_size, features=emb_dim, name='embed')
    x = common_layers.AddPositionEmbs(
        x,
        max_len=max_len,
        posemb_init=common_layers.sinusoidal_init(max_len=max_len),
        cache=cache)
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)
    for _ in range(num_layers):
      x = PerformerBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          causal_mask=True,
          padding_mask=padding_mask,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          cache=cache,
          attention_fn_cls=attention_fn_cls,
          attention_fn_kwargs=attention_fn_kwargs,
      )
    x = nn.LayerNorm(x)
    logits = nn.Dense(
        x,
        vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    return logits
