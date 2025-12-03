import numpy as np
import pytest
from tests.check_gradients import compare_gradients

from tinytorch.activations.relu import ReLU
from tinytorch.activations.sigmoid import Sigmoid
from tinytorch.activations.softmax import Softmax


class TestReLU:
  @pytest.mark.parametrize(
    "shape",
    [
      (3, 5),  # 2D: (B, d)
      (4, 6, 8),  # 3D: (B, T, d)
      (2, 3, 4, 5),  # 4D: (B, H, W, d)
    ],
  )
  def test_gradients_multidim(self, shape: tuple[int, ...]) -> None:
    """ReLU should work with arbitrary input dimensions."""
    module = ReLU()
    x = np.random.randn(*shape)
    assert compare_gradients(module, x)


class TestSigmoid:
  @pytest.mark.parametrize(
    "shape",
    [
      (3, 5),  # 2D: (B, d)
      (4, 6, 8),  # 3D: (B, T, d)
      (2, 3, 4, 5),  # 4D: (B, H, W, d)
    ],
  )
  def test_gradients_multidim(self, shape: tuple[int, ...]) -> None:
    """Sigmoid should work with arbitrary input dimensions."""
    module = Sigmoid()
    x = np.random.randn(*shape)
    assert compare_gradients(module, x)

  def test_gradients_extreme_values(self) -> None:
    """Sigmoid with large positive/negative values (numerical stability)."""
    module = Sigmoid()
    x = np.array([[-10.0, -5.0], [0.0, 5.0], [10.0, 20.0]])
    assert compare_gradients(module, x)


class TestSoftmax:
  @pytest.mark.parametrize(
    "shape",
    [
      (3, 5),  # 2D: (B, d_classes)
      (4, 6, 8),  # 3D: (B, T, d_classes)
      (2, 3, 4, 5),  # 4D: (B, H, W, d_classes)
    ],
  )
  def test_gradients_multidim(self, shape: tuple[int, ...]) -> None:
    """Softmax operates on last dimension, supports arbitrary batch dims."""
    module = Softmax()
    x = np.random.randn(*shape)
    assert compare_gradients(module, x)

  def test_gradients_extreme_values(self) -> None:
    """Softmax with large values (tests numerical stability with exp)."""
    module = Softmax()
    x = np.array([[-100.0, 0.0, 100.0], [50.0, -50.0, 0.0]])
    assert compare_gradients(module, x)
