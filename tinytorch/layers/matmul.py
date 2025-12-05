from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import TwoInputModule
from tinytorch.parameter import Parameter


class MatMul(TwoInputModule):
  def __init__(self) -> None:
    self._x1: np.ndarray | None = None
    self._x2: np.ndarray | None = None

  def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Args:
      x1: shape (..., d1, d2)
      x2: shape (..., d2, d3)

    Output:
      y = x1 @ x2, shape (..., d1, d3)
    """
    self._x1 = x1
    self._x2 = x2
    return x1 @ x2

  def backward(self, grad_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
      grad_out: shape (...)

    Output:
      grad_in1: shape (..., d1, d2).
      grad_in2: shape (..., d2, d3)
    """
    if self._x1 is None or self._x2 is None:
      raise ForwardNotCalledError()

    x1_transpose = self._x1.swapaxes(-1, -2)
    x2_transpose = self._x2.swapaxes(-1, -2)
    return grad_out @ x2_transpose, x1_transpose @ grad_out

  def parameters(self) -> Sequence[Parameter]:
    return []
