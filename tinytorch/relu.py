from typing import Sequence
from tinytorch.error import ForwardNotCalledError
from tinytorch.module import Module
import numpy as np

from tinytorch.parameter import Parameter


class ReLU(Module):
  _x: np.ndarray | None

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    Args:
      x: shape (B, *d)
    """
    self._x = x
    return x * (x > 0)

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    """
    grad_out: shape (B, *d)
    grad_in: shape (B, *d)
    """
    if self._x is None:
      raise ForwardNotCalledError()
    return grad_out * (self._x > 0)

  def parameters(self) -> Sequence[Parameter]:
    return []
