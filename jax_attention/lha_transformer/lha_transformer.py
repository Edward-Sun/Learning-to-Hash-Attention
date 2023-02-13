# Lint as: python3
"""Transformer language models with LHA-based sparse attention."""
import jax.numpy as jnp
from flax import nn

from jax_attention.layers import common_layers
from jax_attention.lha_transformer import mixed_attention


class LHATransformerBlock(nn.Module):
  """LHATransformer layer."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            cache=None,
            n_buckets=10,
            pre_ln=True,
            xl_init=False,
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
            lha_score_func="linear",
            ffn_activation="relu"):
    """Applies MyformerBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: dtype of model
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      cache: flax autoregressive cache for fast decoding.
      n_buckets: int, number of buckets.
      pre_ln: bool
      xl_init: bool
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
      ffn_activation: str

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    if pre_ln:
      x = nn.LayerNorm(inputs)
    else:
      x = inputs

    output = mixed_attention.LHATransformerSelfAttention(
        x,
        num_heads=num_heads,
        qkv_features=qkv_dim,
        causal_mask=causal_mask,
        padding_mask=padding_mask,
        kernel_init=(nn.initializers.normal(stddev=0.02)
                     if xl_init else nn.linear.default_kernel_init),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache,
        n_buckets=n_buckets,
        local_num_heads=local_num_heads,
        local_block_size=local_block_size,
        relative_pos_emb=relative_pos_emb,
        abs_pos_emb_per_layer=abs_pos_emb_per_layer,
        pos_emb=pos_emb,
        r_w_bias=r_w_bias,
        r_r_bias=r_r_bias,
        share_qk_bucket=share_qk_bucket,
        prf_temperature=prf_temperature,
        cluster_score=cluster_score,
        nb_features=nb_features,
        lha_score_func=lha_score_func,
    )
    x, extra_loss = output

    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs

    if not pre_ln:
      x = nn.LayerNorm(x)

    # MLP block.
    if pre_ln:
      y = nn.LayerNorm(x)
    else:
      y = x

    y = common_layers.MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        kernel_init=(nn.initializers.normal(stddev=0.02)
                     if xl_init else nn.linear.default_kernel_init),
        dropout_rate=dropout_rate,
        activation=ffn_activation,
        deterministic=deterministic)
    y = x + y

    if not pre_ln:
      y = nn.LayerNorm(y)

    return y, extra_loss


class LHATransformerEncoder(nn.Module):
  """LHATransformer Model Encoder."""

  def apply(self,
            inputs,
            vocab_size,
            inputs_positions=None,
            inputs_segmentation=None,
            shared_embedding=None,
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
            n_buckets=10,
            pre_ln=True,
            xl_init=False,
            adaptive_embedding=False,
            embedding_dropout_rate=0.0,
            output_dropout_rate=0.0,
            local_num_heads=0,
            local_block_size=0,
            skip_first=0,
            share_qk_bucket=False,
            prf_temperature=1.0,
            cluster_score=0.1):
    """Applies LHATransformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
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
      n_buckets: int, number of buckets.
      pre_ln: bool
      xl_init: bool
      adaptive_embedding: bool
      embedding_dropout_rate: float
      output_dropout_rate: float
      local_num_heads: int
      local_block_size: int
      skip_first: int
      share_qk_bucket: boolean
      prf_temperature: float
      cluster_score: float

    Returns:
      output of a transformer encoder or logits if classifier_mode is true.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[..., None]

    # Input Embedding
    if adaptive_embedding:
      input_embed = common_layers.AdaptiveEmbed.partial(
          n_token=vocab_size,
          d_embed=emb_dim,
          d_proj=emb_dim,
          name='adaptive_embed',
      )
    elif shared_embedding is None:
      input_embed = nn.Embed.partial(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      raise NotImplementedError
    x = inputs.astype('int32')
    x = input_embed(x)

    if classifier and classifier_pool == 'CLS':
      cls = self.param('cls', (1, 1, emb_dim), nn.initializers.zeros)
      cls = jnp.tile(cls, [x.shape[0], 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
      max_len += 1
      src_padding_mask = jnp.concatenate(
          [src_padding_mask[:, :1], src_padding_mask], axis=1)

    if emb_dim != qkv_dim:
      x = nn.linear.DenseGeneral(x, axis=-1, features=qkv_dim)

    pe_init = nn.initializers.normal(stddev=0.02) if learn_pos_emb else None
    x = common_layers.AddPositionEmbs(
        x,
        inputs_positions=inputs_positions,
        posemb_init=pe_init,
        max_len=max_len,
        name='posembed_input')

    if embedding_dropout_rate == 0.0:
      embedding_dropout_rate = dropout_rate
    x = nn.dropout(x, rate=embedding_dropout_rate, deterministic=not train)

    extra_losses = []
    # Input Encoder
    for lyr in range(num_layers):
      x = LHATransformerBlock(
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
          name=f'encoderblock_{lyr}',
          n_buckets=n_buckets,
          pre_ln=pre_ln,
          xl_init=xl_init,
          local_num_heads=num_heads if lyr < skip_first else local_num_heads,
          local_block_size=local_block_size,
          share_qk_bucket=share_qk_bucket,
          prf_temperature=prf_temperature,
          cluster_score=cluster_score,
      )
      x, extra_loss = x
      if extra_loss is not None:
        extra_losses.append(extra_loss)

    if extra_losses:
      extra_losses = jnp.mean(jnp.stack(extra_losses))
    else:
      extra_losses = None

    if pre_ln:
      encoded = nn.LayerNorm(x, name='encoder_norm')
    else:
      encoded = x

    encoded = nn.dropout(encoded, rate=output_dropout_rate,
                         deterministic=not train)

    if classifier:
      encoded = common_layers.classifier_head(
          encoded, num_classes, mlp_dim, pooling_mode=classifier_pool)
    return encoded, extra_losses


class LHATransformerDecoder(nn.Module):
  """LHATransformer Model for language modeling."""

  def apply(self,
            inputs,
            vocab_size,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=True,
            shift=True,
            dtype=jnp.float32,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            cache=None,
            n_buckets=10,
            logits_via_embedding=False,
            adaptive_embedding=False,
            div_val=1,
            proj_same_dim=True,
            cutoffs=None,
            tie_projs=None,
            pre_ln=True,
            relative_pos_emb=False,
            abs_pos_emb_per_layer=False,
            learn_pos_emb=False,
            untie_r=False,
            xl_init=False,
            local_num_heads=0,
            local_block_size=0,
            skip_first=0,
            share_qk_bucket=False,
            prf_temperature=1.0,
            cluster_score=0.1,
            nb_features=256,
            lha_score_func="linear",
            ffn_activation="relu"):
    """Applies LHATransformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: bool: if model is training.
      shift: bool: if we right-shift input - this is only disabled for
        fast, looped single-token autoregressive decoding.
      dtype: the dtype of the computation (default: float32)
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      cache: flax autoregressive cache for fast decoding.
      n_buckets: int
      logits_via_embedding: boolean
      adaptive_embedding: boolean
      div_val: integer
      proj_same_dim: boolean
      cutoffs: list
      tie_projs: list
      pre_ln: boolean
      relative_pos_emb: boolean
      abs_pos_emb_per_layer: boolean
      learn_pos_emb: boolean
      untie_r: boolean
      xl_init: boolean
      local_num_heads: int
      local_block_size: int
      skip_first: int
      share_qk_bucket: boolean
      prf_temperature: float
      cluster_score: float
      nb_features: int
      lha_score_func: str
      ffn_activation: str

    Returns:
      output of a myformer decoder.
    """
    padding_mask = (inputs > 0)[..., None]
    assert inputs.ndim == 2  # (batch, len)
    x = inputs
    if shift:
      x = common_layers.shift_right(x)
    x = x.astype('int32')

    shared_embed = None
    if adaptive_embedding:
      shared_embed = common_layers.AdaptiveEmbed.shared(
          n_token=vocab_size,
          d_embed=emb_dim,
          d_proj=emb_dim,
          cutoffs=cutoffs,
          initializer=nn.initializers.normal(stddev=0.02),
          proj_initializer=nn.initializers.normal(stddev=0.01),
          tie_projs=tie_projs,
          div_val=div_val,
          proj_same_dim=proj_same_dim,
          name='adaptive_embed',
      )
      x = shared_embed(x)
    elif logits_via_embedding:
      shared_embed = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          name='embed',
          embedding_init=nn.initializers.normal(stddev=1.0))
      x = shared_embed(x)
    else:
      x = common_layers.Embed(
          x, num_embeddings=vocab_size, features=emb_dim, name='embed')

    pos_emb = None
    r_w_bias = None
    r_r_bias = None

    if relative_pos_emb or abs_pos_emb_per_layer:
      if learn_pos_emb:
        pos_emb_init = nn.initializers.normal(stddev=0.02)
      else:
        pos_emb_init = common_layers.sinusoidal_init(max_len=max_len)
      pos_emb_shape = (1, max_len, emb_dim)
      pos_emb = self.param('pos_embedding', pos_emb_shape, pos_emb_init)
      pos_emb = jnp.reshape(pos_emb, (max_len, emb_dim))
      pos_emb = pos_emb[::-1]
      pos_emb = nn.dropout(pos_emb, rate=dropout_rate, deterministic=not train)
      x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

      if relative_pos_emb:
        d_head = qkv_dim // num_heads

        if untie_r:
          r_w_bias = self.param('r_w_bias', (num_layers, num_heads, d_head),
                                initializer=nn.initializers.normal(stddev=0.02))
          r_r_bias = self.param('r_r_bias', (num_layers, num_heads, d_head),
                                initializer=nn.initializers.normal(stddev=0.02))
        else:
          r_w_bias = self.param('r_w_bias', (num_heads, d_head),
                                initializer=nn.initializers.normal(stddev=0.02))
          r_r_bias = self.param('r_r_bias', (num_heads, d_head),
                                initializer=nn.initializers.normal(stddev=0.02))

    else:
      pe_init = nn.initializers.normal(stddev=0.02) if learn_pos_emb else None

      x = common_layers.AddPositionEmbs(
          x,
          max_len=max_len,
          posemb_init=pe_init,
          cache=cache)
      x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    extra_losses = []
    for lyr in range(num_layers):
      x = LHATransformerBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          causal_mask=True,
          dtype=dtype,
          padding_mask=padding_mask,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          cache=cache,
          n_buckets=n_buckets,
          pre_ln=pre_ln,
          xl_init=xl_init,
          relative_pos_emb=relative_pos_emb,
          abs_pos_emb_per_layer=abs_pos_emb_per_layer,
          pos_emb=pos_emb,
          r_w_bias=r_w_bias if not untie_r  else r_w_bias[lyr],
          r_r_bias=r_r_bias if not untie_r  else r_r_bias[lyr],
          local_num_heads=num_heads if lyr < skip_first else local_num_heads,
          local_block_size=local_block_size,
          share_qk_bucket=share_qk_bucket,
          prf_temperature=prf_temperature,
          cluster_score=cluster_score,
          nb_features=nb_features,
          lha_score_func=lha_score_func,
          ffn_activation=ffn_activation,
          name=f'decoderblock_{lyr}',
      )

      x, extra_loss = x

      if extra_loss is not None:
        extra_losses.append(extra_loss)

    if extra_losses:
      extra_losses = jnp.mean(jnp.stack(extra_losses))
    else:
      extra_losses = None

    if pre_ln:
      x = nn.LayerNorm(x, name="pre_ln_final")

    bias_init = nn.initializers.normal(stddev=1e-6)

    if adaptive_embedding:
      logits = shared_embed.attend(x.astype(jnp.float32))
      logit_bias = self.param('embed_bias', (vocab_size, ), bias_init)
      logits = logits + logit_bias

    elif logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = shared_embed.attend(x.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(x.shape[-1])
      logit_bias = self.param('embed_bias', (vocab_size, ), bias_init)
      logits = logits + logit_bias

    else:
      logits = nn.Dense(
          x.astype(jnp.float32),
          vocab_size,
          name="dense_logits",
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=bias_init)

    return logits, extra_losses
