import numpy as np
import pytest

from tests.check_gradients import compare_gradients
from tinytorch.transformers.transformer_block import TransformerBlock


class TestTransformerBlock:
  @pytest.mark.parametrize(
    "batch,seq_len,d_model,num_heads,d_ff",
    [
      (1, 4, 8, 2, 16),
      (2, 6, 16, 4, 32),
    ],
  )
  def test_forward_shape(
    self, batch: int, seq_len: int, d_model: int, num_heads: int, d_ff: int
  ) -> None:
    block = TransformerBlock(num_heads, d_model, d_ff, is_causal=True)
    x = np.random.randn(batch, seq_len, d_model)
    out = block.forward(x)
    assert out.shape == (batch, seq_len, d_model)

  @pytest.mark.parametrize(
    "batch,seq_len,d_model,num_heads,d_ff",
    [
      (1, 3, 8, 2, 16),
      (2, 4, 8, 2, 16),
    ],
  )
  def test_gradient(
    self, batch: int, seq_len: int, d_model: int, num_heads: int, d_ff: int
  ) -> None:
    block = TransformerBlock(num_heads, d_model, d_ff, is_causal=True)
    x = np.random.randn(batch, seq_len, d_model) * 0.1
    block.zero_grad()
    assert compare_gradients(block, x)

  def test_residual_connection(self) -> None:
    """Output should change if input changes (residual passes through)."""
    block = TransformerBlock(num_heads=2, d_model=8, d_ff=16, is_causal=True)
    x1 = np.random.randn(1, 4, 8)
    x2 = x1 + 0.1
    out1 = block.forward(x1)
    out2 = block.forward(x2)
    assert not np.allclose(out1, out2)
