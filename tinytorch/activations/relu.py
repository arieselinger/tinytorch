from typing import Sequence
from tinytorch.exceptions import ForwardNotCalledError
import numpy as np

from tinytorch.parameter import Parameter
from tinytorch.module import OneInputModule


class ReLU(OneInputModule):
  _x: np.ndarray | None

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    Args:
      x: shape (...)
    """
    self._x = x
    return x * (x > 0)

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    """
    Args:
      grad_out: shape (...)

    Output:
      grad_in: shape (...)
    """
    if self._x is None:
      raise ForwardNotCalledError()
    return grad_out * (self._x > 0)

  def parameters(self) -> Sequence[Parameter]:
    return []
