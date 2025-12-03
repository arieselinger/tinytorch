import numpy as np
import pytest

from tests.check_gradients import compare_gradients
from tinytorch.layers.layer_norm import LayerNorm


class TestLayerNormGradients:
  """Test gradient correctness using numerical gradient checking."""

  @pytest.mark.parametrize(
    "batch_size,d_in",
    [
      (1, 5),  # Single sample
      (2, 3),  # Small batch
      (4, 10),  # Medium batch
      (8, 50),  # Large batch with realistic dimensions
    ],
  )
  def test_2d_inputs(self, batch_size: int, d_in: int) -> None:
    """Test standard 2D batch inputs: (batch_size, d_in)."""
    module = LayerNorm(d_in)
    x = np.random.randn(batch_size, d_in) * 5 + 10
    assert compare_gradients(module, x)

  @pytest.mark.parametrize(
    "shape",
    [
      ((2, 3, 4)),  # (B, T, d_in)
      ((2, 3, 4, 5)),  # (B, T, H, d_in)
      ((3, 2, 4, 3, 8)),  # (B, C, T, H, d_in)
    ],
  )
  def test_multidim_inputs(self, shape: tuple[int, ...]) -> None:
    """Test arbitrary batch dimensions."""
    d_in = shape[-1]
    module = LayerNorm(d_in)
    x = np.random.randn(*shape)
    assert compare_gradients(module, x)

  @pytest.mark.parametrize(
    "d_in,distribution,batch_size",
    [
      (10, "uniform", 4),
      (20, "uniform", 4),
      (10, "exponential", 4),
      (20, "exponential", 4),
      (10, "skewed", 4),
      (20, "skewed", 4),
    ],
  )
  def test_non_normal_distributions(self, d_in: int, distribution: str, batch_size: int) -> None:
    """Test with various non-normal distributions."""
    module = LayerNorm(d_in)

    if distribution == "uniform":
      x = np.random.uniform(0, 1, (batch_size, d_in))
    elif distribution == "exponential":
      x = np.random.exponential(2.0, (batch_size, d_in))
    else:  # skewed
      x = np.random.randn(batch_size, d_in) ** 3

    assert compare_gradients(module, x)


class TestLayerNormForward:
  """Test forward pass properties."""

  @pytest.mark.parametrize(
    "batch_size,d_in,scale,offset",
    [
      (5, 5, 3, 7),
      (5, 10, 3, 7),
      (5, 20, 3, 7),
      (5, 50, 3, 7),
    ],
  )
  def test_output_normalization(
    self, batch_size: int, d_in: int, scale: float, offset: float
  ) -> None:
    """Verify normalized output has mean≈0 and std≈1 per position."""
    module = LayerNorm(d_in)
    x = np.random.randn(batch_size, d_in) * scale + offset

    y = module.forward(x)

    mean_per_row = np.mean(y, axis=-1)
    std_per_row = np.std(y, axis=-1)

    assert np.allclose(mean_per_row, 0, atol=1e-6)
    assert np.allclose(std_per_row, 1, atol=1e-6)

  @pytest.mark.parametrize(
    "d_in",
    [5, 10, 20],
  )
  def test_parameter_initialization(self, d_in: int) -> None:
    """Verify gamma initialized to 1 and beta to 0."""
    module = LayerNorm(d_in)

    params = module.parameters()
    assert len(params) == 2
    assert params[0].data.shape == (d_in,)  # gamma
    assert params[1].data.shape == (d_in,)  # beta
    assert np.allclose(params[0].data, 1.0)  # gamma = 1
    assert np.allclose(params[1].data, 0.0)  # beta = 0
