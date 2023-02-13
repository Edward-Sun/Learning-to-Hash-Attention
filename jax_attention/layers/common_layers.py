"""Common layers used in models."""
import jax.numpy as jnp
import numpy as np
from flax import nn
from jax import lax


def rel_shift(x):
  x_size = x.shape
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[1] = (1, 0)  # Padding on axis=1
  x = jnp.pad(x, pad_widths, mode='constant', constant_values=x.dtype.type(0))

  if len(x.shape) == 4:
    x = jnp.reshape(x, (x.shape[1], x.shape[0], x.shape[2], x.shape[3]))
    x = x[1:, :, :, :]
    x = jnp.reshape(x, x_size)
  else:
    raise NotImplementedError

  return x


def shift_right(x):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[1] = (1, 0)  # Padding on axis=1
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[:, :-1]


class MlpBlock(nn.Module):
  """Transformer MLP block."""

  def apply(self,
            inputs,
            mlp_dim,
            dtype=jnp.float32,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=False,
            activation="gelu",
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    inputs = inputs.astype(dtype)
    x = nn.Dense(inputs, mlp_dim, dtype=dtype,
                 kernel_init=kernel_init, bias_init=bias_init)
    if activation == "relu":
      x = nn.relu(x)
    elif activation == "gelu":
      x = nn.gelu(x)
    elif activation == "swish":
      x = nn.swish(x)
    elif activation == "relu_squared":
      x = jnp.square(nn.relu(x))
    elif activation == "swiglu":
      x_a, x_b = jnp.split(x, 2, axis=-1)
      x = x_a * nn.swish(x_b)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    output = nn.Dense(
        x, actual_out_dim, dtype=dtype, kernel_init=kernel_init,
        bias_init=bias_init)
    output = nn.dropout(output, rate=dropout_rate, deterministic=deterministic)
    output = output.astype(jnp.float32)
    return output


def classifier_head(encoded, num_classes, mlp_dim, pooling_mode='MEAN'):
  """Classifier head.

  We put this here just so that all models consistently call the same function.

  Args:
    encoded: tensor inputs are shape of [bs, len, dim].
    num_classes: int, number of classes
    mlp_dim: int, dim of intermediate MLP.
    pooling_mode: str, string dictating pooling op {MEAN}

  Returns:
    tensor of shape [bs, num_classes]

  """
  if pooling_mode == 'MEAN':
    encoded = jnp.mean(encoded, axis=1)
  elif pooling_mode == 'SUM':
    encoded = jnp.sum(encoded, axis=1)
  elif pooling_mode == 'FLATTEN':
    encoded = encoded.reshape((encoded.shape[0], -1))
  elif pooling_mode == 'CLS':
    encoded = encoded[:, 0]
  else:
    raise NotImplementedError('Pooling not supported yet.')
  encoded = nn.Dense(encoded, mlp_dim, name='mlp')
  encoded = nn.relu(encoded)
  encoded = nn.Dense(encoded, num_classes, name='logits')
  return encoded


class Embed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self,
            inputs,
            num_embeddings,
            features,
            mode='input',
            emb_init=nn.initializers.normal(stddev=1.0)):
    """Applies Embed module.

    Args:
      inputs: input data
      num_embeddings: number of embedding
      features: size of the embedding dimension
      mode: either 'input' or 'output' -> to share input/output embedding
      emb_init: embedding initializer

    Returns:
      output which is embedded input data
    """
    embedding = self.param('embedding', (num_embeddings, features), emb_init)
    if mode == 'input':
      if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError('Input type must be an integer or unsigned integer.')
      return jnp.take(embedding, inputs, axis=0)
    if mode == 'output':
      return jnp.einsum('bld,vd->blv', inputs, embedding)


class AdaptiveEmbed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self, x, n_token, d_embed, d_proj, cutoffs=None,
            initializer=nn.initializers.normal(stddev=0.02),
            proj_initializer=nn.initializers.normal(stddev=0.01),
            div_val=1, proj_same_dim=True):
    """
    perms: If None, first compute W = W1 x W2 (projection for each bin),
        and then compute X x W (embedding lookup). If not None,
        use bin-based embedding lookup with max_bin_size defined by
        the shape of perms.
    """
    emb_scale = d_proj ** 0.5

    if div_val == 1:
      lookup_table = self.param('lookup_table',
                                (n_token, d_embed),
                                initializer)
      y = lookup_table[x]
      if d_proj != d_embed:
        proj_W = self.param('proj_W',
                            (d_embed, d_proj),
                            proj_initializer)
        y = jnp.einsum('ibe,ed->ibd', y, proj_W)
    else:
      cutoff_ends = [0] + cutoffs + [n_token]
      cat_lookup = []

      for i in range(len(cutoff_ends) - 1):
        l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
        cur_d_embed = d_embed // (div_val ** i)
        lookup_table = self.param('lookup_table_%d' % i,
                                  (r_idx - l_idx, cur_d_embed),
                                  initializer)
        if cur_d_embed != d_proj or proj_same_dim:
          proj_W = self.param('proj_W_%d' % i,
                              (cur_d_embed, d_proj),
                              proj_initializer)
          cat_lookup.append(jnp.einsum('ie,ed->id', lookup_table, proj_W))
        else:
          cat_lookup.append(lookup_table)

      cat_lookup = jnp.concatenate(cat_lookup, 0)
      y = cat_lookup[x]

    y *= emb_scale
    return y

  @nn.base.module_method
  def attend(self, query, **kwargs):
    div_val = kwargs['div_val']
    d_proj = kwargs['d_proj']
    d_embed = kwargs['d_embed']
    tie_projs = kwargs['tie_projs']
    cutoffs = kwargs['cutoffs']
    n_token = kwargs['n_token']
    proj_same_dim = kwargs['proj_same_dim']

    if div_val == 1:
      lookup_table = self.get_param('lookup_table')

      if d_proj != d_embed:
        if tie_projs is None or not tie_projs[0]:
          proj_W = self.get_param('rev_proj_W')
        else:
          proj_W = self.get_param('proj_W')
        embedding = jnp.einsum('ie,ed->id', lookup_table, proj_W)
      else:
        embedding = lookup_table

    else:
      cutoff_ends = [0] + cutoffs + [n_token]
      cat_lookup = []

      for i in range(len(cutoff_ends) - 1):
        cur_d_embed = d_embed // (div_val ** i)
        lookup_table = self.get_param('lookup_table_%d' % i)

        if cur_d_embed != d_proj or proj_same_dim:
          if tie_projs is None or not tie_projs[i]:
            proj_W = self.get_param('rev_proj_W_%d' % i)
          else:
            proj_W = self.get_param('proj_W_%d' % i)

          cat_lookup.append(jnp.einsum('ie,ed->id', lookup_table, proj_W))
        else:
          cat_lookup.append(lookup_table)

      embedding = jnp.concatenate(cat_lookup, 0)

    return lax.dot_general(
        query, embedding, (((query.ndim - 1,), (1,)), ((), ())))


def sinusoidal_init(max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs."""

  def apply(self,
            inputs,
            inputs_positions=None,
            max_len=512,
            posemb_init=None,
            cache=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.
      max_len: maximum possible length for the input.
      posemb_init: positional embedding initializer, if None, then use a
        fixed (non-learned) sinusoidal embedding table.
      cache: flax attention cache for fast decoding.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, max_len, inputs.shape[-1])
    if posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(
          max_len=max_len)(None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding', pos_emb_shape, posemb_init)
    pe = pos_embedding[:, :length, :]
    # We abuse the same attention Cache mechanism to run positional embeddings
    # in fast predict mode. We could use state variables instead, but this
    # simplifies invocation with a single top-level cache context manager.
    # We only use the cache's position index for tracking decoding position.
    if cache:
      if self.is_initializing():
        cache.store(np.array((4, 1, 1), dtype=np.int32))
      else:
        cache_entry = cache.retrieve(None)
        i = cache_entry.i
        cache.store(cache_entry.replace(i=cache_entry.i + 1))
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)
