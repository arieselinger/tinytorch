from typing import Sequence
import numpy as np
from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import OneInputModule
from tinytorch.parameter import Parameter


class Softmax(OneInputModule):
  _s: np.ndarray | None

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    Compute softmax along last dimension

    Args
      x: shape (..., d_classes)

    Output
      softmax(x): shape (..., d_classes)

    Notes
      1. Softmax encodes relative positions since each vector can be shifted by a constant c:
      exp(x) / sum{exp(x)} = exp(x-c) / sum{exp(x-c)}
      2. For numerical stability, we compute c = max(x)
    """
    x_max = np.max(x, axis=-1, keepdims=True)  # shape (..., 1)
    exp_x = np.exp(x - x_max)  # shape (..., d_classes)
    s = exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # shape (..., d_classes)

    self._s = s
    return s

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    """
    Args:
      grad_out: shape (..., d_classes)
      self._s: shape (..., d_classes)

    ----

    s[j] = exp(x[j]) / np.sum(exp(x))

    Jacobian:
      J_ji
      = ds[j]/dx[i]
      = s[i]*(d_ij - s[j])

    grad_in[j]
      = (J.T @ grad_out)[j]
      = J_ji g[i]
      = (s[i]*(d_ij - s[j])) g[i]
      = s[i]*dij*g[i] - s[i]s[j]g[i]
      = s[j]g[j] - s[j]<s,g>
      = s[j] * (g[j] - <s,g>)
    """
    if self._s is None:
      raise ForwardNotCalledError()
    dot = np.sum(self._s * grad_out, axis=-1, keepdims=True)  # shape (..., 1)
    return self._s * (grad_out - dot)

  def parameters(self) -> Sequence[Parameter]:
    return []
