from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import OneInputModule
from tinytorch.parameter import Parameter


class GELU(OneInputModule):
  """
  Gaussian Error Linear Unit activation function
  Using the tanh approximation

  d(GELU)/dx = 0.5 * [1 + tanh(g) + x * sech^2(g) * g'(x)]
  """

  _x: np.ndarray | None
  _tanh_g: np.ndarray | None

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    Args:
      x: shape (...)
    """
    self._x = x
    g: np.ndarray = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    tanh_g: np.ndarray = np.tanh(g)
    self._tanh_g = tanh_g
    return 0.5 * x * (1 + tanh_g)

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    """
    Args:
      grad_out: shape (...)

    Output:
      grad_in: shape (...)
    """
    if self._x is None or self._tanh_g is None:
      raise ForwardNotCalledError()

    # Derivatives
    sech2_g = 1 - self._tanh_g**2
    g_prime = np.sqrt(2 / np.pi) * (1 + 0.134145 * self._x**2)

    # d(GELU)/dx = 0.5 * [1 + tanh(g) + x * sech^2(g) * g'(x)]
    grad_x = 0.5 * (1 + self._tanh_g + self._x * sech2_g * g_prime) * grad_out

    return grad_x

  def parameters(self) -> Sequence[Parameter]:
    return []
