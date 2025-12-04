from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import OneInputModule
from tinytorch.parameter import Parameter


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
