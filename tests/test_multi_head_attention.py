import numpy as np
import pytest

from tests.check_gradients import compare_gradients
from tinytorch.transformers.multi_head_attention import MultiHeadAttention


@pytest.mark.parametrize(
  "num_heads,d_model,shape",
  [
    (1, 8, (1, 2, 8)),  # single head
    (2, 8, (2, 3, 8)),  # batched 3D
    (2, 8, (3, 8)),  # 2D no batch
    (2, 8, (2, 2, 3, 8)),  # 4D extra dims
    (8, 8, (1, 2, 8)),  # many heads (num_heads = d_model)
  ],
)
def test_output_shape(num_heads: int, d_model: int, shape: tuple[int, ...]) -> None:
  """Output shape matches input shape for various dimensions."""
  mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
  x = np.random.randn(*shape)
  output = mha.forward(x)
  assert output.shape == shape

  grad_out = np.random.randn(*shape)
  grad_in = mha.backward(grad_out)
  assert grad_in.shape == shape


def test_invalid_d_model() -> None:
  """Should raise error if d_model is not divisible by num_heads."""
  with pytest.raises(ValueError, match="d_model should be a multiple of num_heads"):
    MultiHeadAttention(num_heads=4, d_model=63)


@pytest.mark.parametrize(
  "num_heads,d_model,shape",
  [
    (2, 8, (1, 2, 8)),  # 3D batched
    (2, 8, (3, 8)),  # 2D no batch
  ],
)
def test_gradients(num_heads: int, d_model: int, shape: tuple[int, ...]) -> None:
  """Gradients are numerically correct."""
  mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model, causal_mask=False)
  x = np.random.randn(*shape)
  assert compare_gradients(mha, x)
