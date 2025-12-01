import numpy as np
import pytest

from tests.check_gradients import compare_gradients
from tinytorch.layers.linear import Linear
from tinytorch.exceptions import ForwardNotCalledError


class TestLinearGradients:
  """Test gradient correctness using numerical gradient checking."""

  @pytest.mark.parametrize(
    "batch_size,d_in,d_out",
    [
      (1, 5, 3),  # Single sample
      (2, 3, 5),  # Small batch
      (4, 10, 20),  # Medium batch
      (8, 50, 100),  # Large batch with realistic dimensions
    ],
  )
  def test_2d_inputs(self, batch_size: int, d_in: int, d_out: int) -> None:
    """Test standard 2D batch inputs: (batch_size, d_in)."""
    module = Linear(d_in, d_out)
    x = np.random.randn(batch_size, d_in)
    assert compare_gradients(module, x)

  @pytest.mark.parametrize(
    "batch_size,d_in,d_out",
    [
      (4, 10, 5),
      (8, 20, 15),
    ],
  )
  def test_no_bias(self, batch_size: int, d_in: int, d_out: int) -> None:
    """Test gradient correctness when bias is disabled."""
    module = Linear(d_in, d_out, bias=False)
    x = np.random.randn(batch_size, d_in)
    assert compare_gradients(module, x)

  @pytest.mark.parametrize(
    "shape,d_out",
    [
      ((2, 3, 4), 3),  # 3D: (B, T, d_in) - sequence/time series
      ((2, 3, 4, 5), 10),  # 4D: (B, T, H, d_in) - spatial-temporal
      ((3, 2, 4, 3, 8), 6),  # 5D: (B, C, T, H, d_in) - multi-channel temporal
    ],
  )
  def test_multidim_inputs(
    self,
    shape: tuple[int, ...],
    d_out: int,
  ) -> None:
    """Test arbitrary batch dimensions."""
    d_in = shape[-1]
    module = Linear(d_in, d_out)
    x = np.random.randn(*shape)
    assert compare_gradients(module, x)


class TestLinearBackward:
  """Test backward pass functionality and gradient computation."""

  def test_error_without_forward(self) -> None:
    """Backward should raise error if called before forward (needs cached input)."""
    module = Linear(5, 3)
    grad_out = np.random.randn(2, 3)
    with pytest.raises(ForwardNotCalledError):
      module.backward(grad_out)


class TestLinearParameters:
  """Test parameter management and initialization."""

  def test_parameters_with_bias(self) -> None:
    """Parameters() should return [W, b] when bias=True."""
    module = Linear(5, 3, bias=True)
    params = module.parameters()
    assert len(params) == 2
    assert params[0] is module.W
    assert params[1] is module.b

  def test_parameters_no_bias(self) -> None:
    """Parameters() should return [W] when bias=False."""
    module = Linear(5, 3, bias=False)
    params = module.parameters()
    assert len(params) == 1
    assert params[0] is module.W
    assert module.b is None
