import numpy as np
import pytest

from tests.check_gradients import compare_three_input_gradients
from tinytorch.transformers.scaled_dot_product_attention import ScaledDotProductAttention


@pytest.mark.parametrize(
  "seq_len,d_k,d_v",
  [
    (1, 2, 2),
    (3, 2, 4),
    (5, 8, 8),
    (10, 16, 16),
  ],
)
def test_square_attention_shape(seq_len: int, d_k: int, d_v: int) -> None:
  """Output has correct shape for square attention (num_queries=num_queries=seq_len)."""
  module = ScaledDotProductAttention(is_causal=True)

  q = np.random.randn(1, seq_len, d_k)
  k = np.random.randn(1, seq_len, d_k)
  v = np.random.randn(1, seq_len, d_v)

  out = module.forward(q, k, v)

  # Output shape should be (batch, seq_len, d_v)
  assert out.shape == (1, seq_len, d_v)


@pytest.mark.parametrize(
  "batch_size,num_queries,seq_len,d_k,d_v",
  [
    (1, 1, 3, 2, 2),
    (1, 2, 5, 3, 4),
    (1, 5, 10, 8, 8),
    (3, 10, 20, 16, 16),
  ],
)
def test_inference_shape(
  batch_size: int, num_queries: int, seq_len: int, d_k: int, d_v: int
) -> None:
  """Output has correct shape for non-square attention (num_queries<num_keys, KV cache)."""
  module = ScaledDotProductAttention(is_causal=True)

  q = np.random.randn(batch_size, num_queries, d_k)
  k = np.random.randn(batch_size, seq_len, d_k)
  v = np.random.randn(batch_size, seq_len, d_v)

  out = module.forward(q, k, v)

  # Output shape should be (batch, num_queries, d_v)
  assert out.shape == (batch_size, num_queries, d_v)


def test_is_causal_affects_output() -> None:
  """Causal mask produces different output than non-causal (early positions see fewer keys)."""
  module_causal = ScaledDotProductAttention(is_causal=True)
  module_no_causal = ScaledDotProductAttention(is_causal=False)

  rng = np.random.default_rng(123)
  q = rng.standard_normal((1, 3, 2))
  k = rng.standard_normal((1, 3, 2))
  v = rng.standard_normal((1, 3, 2))

  out_causal = module_causal.forward(q, k, v)
  out_no_causal = module_no_causal.forward(q, k, v)

  # Early positions should see fewer keys with causal mask so outputs should differ
  assert not np.allclose(out_causal[0, 0, :], out_no_causal[0, 0, :], atol=1e-4)


def test_attention_weights_sum_values() -> None:
  """Output is weighted sum of values (with identical keys, weights are uniform)."""
  module = ScaledDotProductAttention(is_causal=False)

  q = np.array([[[1.0, 0.0]]])  # shape (1, 1, 2)
  k = np.array([[[1.0, 0.0], [1.0, 0.0]]])  # shape (1, 2, 2) - identical keys
  v = np.array([[[100.0], [200.0]]])  # shape (1, 2, 1)

  out = module.forward(q, k, v)

  # With identical keys, attention weights should be uniform [0.5, 0.5]
  # Output = 0.5 * 100 + 0.5 * 200 = 150
  assert np.allclose(out[0, 0, 0], 150.0, atol=1e-3)


# GRADIENT CORRECTNESS TESTS


@pytest.mark.parametrize(
  "seq_len,d_k",
  [
    (2, 2),
    (3, 4),
    (5, 8),
    (8, 16),
  ],
)
def test_gradients_square_causal(seq_len: int, d_k: int) -> None:
  """Gradients correct for square attention with causal masking."""
  module = ScaledDotProductAttention(is_causal=True)

  rng = np.random.default_rng(42)
  q = rng.standard_normal((1, seq_len, d_k))
  k = rng.standard_normal((1, seq_len, d_k))
  v = rng.standard_normal((1, seq_len, d_k))

  assert compare_three_input_gradients(module, q, k, v)


@pytest.mark.parametrize(
  "num_queries,seq_len,d_k",
  [
    (1, 3, 2),
    (2, 5, 3),
    (5, 10, 8),
  ],
)
def test_gradients_inference_causal(num_queries: int, seq_len: int, d_k: int) -> None:
  """Gradients correct for inference scenario (single query, causal mask)."""
  module = ScaledDotProductAttention(is_causal=True)

  rng = np.random.default_rng(42)
  q = rng.standard_normal((1, num_queries, d_k))
  k = rng.standard_normal((1, seq_len, d_k))
  v = rng.standard_normal((1, seq_len, d_k))

  assert compare_three_input_gradients(module, q, k, v)


@pytest.mark.parametrize(
  "seq_len,d_k",
  [
    (2, 2),
    (3, 4),
    (5, 8),
  ],
)
def test_gradients_square_no_causal(seq_len: int, d_k: int) -> None:
  """Gradients correct for square attention without causal masking."""
  module = ScaledDotProductAttention(is_causal=False)

  rng = np.random.default_rng(42)
  q = rng.standard_normal((1, seq_len, d_k))
  k = rng.standard_normal((1, seq_len, d_k))
  v = rng.standard_normal((1, seq_len, d_k))

  assert compare_three_input_gradients(module, q, k, v)
