import numpy as np
from typing import Sequence
from tinytorch.error import ForwardNotCalledError
from tinytorch.layers.module import Module
from tinytorch.parameter import Parameter


class MSELoss(Module):
  _diff: np.ndarray | None
  _n: int | None

  def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
    """
    Output:
      <X-Y,X-Y>_F / N = tr((X-Y).T @ (X-Y)) / N

    Gradient:
      grad_X =
    """
    if y_pred.shape != y_true.shape:
      raise RuntimeError("y_true and y_pred should have the same shapes")
    self._n = y_pred.size
    self._diff = y_pred - y_true
    return np.sum(self._diff**2) / self._n

  def backward(self, grad_out: np.floating = np.float64(1.0)) -> np.ndarray:
    if self._n is None or self._diff is None:
      raise ForwardNotCalledError()
    return grad_out * 2 * self._diff / self._n

  def parameters(self) -> Sequence[Parameter]:
    return []
