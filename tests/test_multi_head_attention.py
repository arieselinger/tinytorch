import numpy as np
import pytest

from tests.check_gradients import compare_gradients
from tinytorch.transformers.multi_head_attention import MultiHeadAttention


@pytest.mark.parametrize(
  "num_heads,d_model,b,t",
  [
    (1, 16, 1, 1),
    (2, 64, 2, 5),
    (4, 64, 3, 10),
    (8, 256, 2, 20),
  ],
)
def test_output_shape(num_heads: int, d_model: int, b: int, t: int) -> None:
  """Output shape matches input shape."""
  mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
  x = np.random.randn(b, t, d_model)
  output = mha.forward(x)
  assert output.shape == x.shape


def test_invalid_d_model() -> None:
  """Should raise error if d_model is not divisible by num_heads."""
  with pytest.raises(ValueError, match="d_model should be a multiple of num_heads"):
    MultiHeadAttention(num_heads=4, d_model=63)


@pytest.mark.parametrize(
  "num_heads,d_model,b,t",
  [
    (2, 32, 1, 2),
    (4, 64, 2, 3),
  ],
)
def test_gradients(num_heads: int, d_model: int, b: int, t: int) -> None:
  """Gradients are numerically correct."""
  mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model, causal_mask=False)
  x = np.random.randn(b, t, d_model)
  assert compare_gradients(mha, x)


def test_single_head() -> None:
  """Single head (num_heads=1) should work correctly."""
  mha = MultiHeadAttention(num_heads=1, d_model=32, causal_mask=True)
  x = np.random.randn(2, 5, 32)
  output = mha.forward(x)
  assert output.shape == x.shape

  grad_out = np.random.randn(2, 5, 32)
  grad_in = mha.backward(grad_out)
  assert grad_in.shape == x.shape


def test_many_heads() -> None:
  """Many heads (num_heads = d_model) should work correctly."""
  mha = MultiHeadAttention(num_heads=64, d_model=64, causal_mask=False)
  x = np.random.randn(1, 3, 64)
  output = mha.forward(x)
  assert output.shape == x.shape

  grad_out = np.random.randn(1, 3, 64)
  grad_in = mha.backward(grad_out)
  assert grad_in.shape == x.shape
