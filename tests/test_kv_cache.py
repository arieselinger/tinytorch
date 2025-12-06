"""Test KVCache with autoregressive forward pass."""

import numpy as np

from tinytorch.transformers.kv_cache import KVCache
from tinytorch.transformers.multi_head_attention import MultiHeadAttention


def test_kv_cache_single_step():
  """Test cache with single new token."""
  np.random.seed(42)

  B, T, d_model = 2, 5, 64
  num_heads = 4
  mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model, causal_mask=True)
  cache = KVCache()

  # Full sequence forward with cache
  x_full = np.random.randn(B, T, d_model)
  mha.forward(x_full, cache=cache)

  # New token with cache
  x_new = np.random.randn(B, 1, d_model)
  output_new_cache = mha.forward(x_new, cache=cache)

  # Full sequence forward (no cache) for comparison
  cache.reset_all()
  x_concat = np.concatenate([x_full, x_new], axis=1)
  output_concat = mha.forward(x_concat)

  # Compare: cache output should match full sequence
  output_cache_last = output_new_cache[:, 0, :]
  output_full_last = output_concat[:, -1, :]
  diff = np.max(np.abs(output_cache_last - output_full_last))

  assert np.allclose(output_cache_last, output_full_last, atol=1e-6), f"Mismatch: {diff}"


def test_kv_cache_multiple_steps():
  """Test cache with multiple new tokens."""
  np.random.seed(42)

  B, T, d_model = 2, 4, 64
  num_heads = 4
  mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model, causal_mask=True)
  cache = KVCache()

  # Initial sequence
  x_initial = np.random.randn(B, T, d_model)
  mha.forward(x_initial, cache=cache)

  # Generate 3 new tokens with cache
  all_inputs = [x_initial]
  output_cached: np.ndarray | None = None
  for _ in range(3):
    x_new = np.random.randn(B, 1, d_model)
    output_cached = mha.forward(x_new, cache=cache)
    all_inputs.append(x_new)

  # Compare with full forward
  cache.reset_all()
  x_full = np.concatenate(all_inputs, axis=1)
  output_full = mha.forward(x_full)

  # Last token should match
  assert output_cached is not None
  output_cache_last = output_cached[:, 0, :]
  output_full_last = output_full[:, -1, :]
  diff = np.max(np.abs(output_cache_last - output_full_last))

  assert np.allclose(output_cache_last, output_full_last, atol=1e-6), f"Mismatch: {diff}"
