from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import CriterionModule
from tinytorch.parameter import Parameter


class MSELoss(CriterionModule):
  _diff: np.ndarray | None
  _size: int | None

  def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Output:
      L(x,y) = 1 / N * <X-Y,X-Y>_F

    Gradient:
      grad_X = 2 / N * (X - Y)
    """
    if y_pred.shape != y_true.shape:
      raise RuntimeError("y_true and y_pred should have the same shapes")
    self._size = y_pred.size
    self._diff = y_pred - y_true
    return np.array(np.sum(self._diff**2) / self._size)

  def backward(self, grad_out: np.ndarray = np.array(1.0)) -> np.ndarray:
    if self._size is None or self._diff is None:
      raise ForwardNotCalledError()
    return grad_out * 2 * self._diff / self._size

  def parameters(self) -> Sequence[Parameter]:
    return []
