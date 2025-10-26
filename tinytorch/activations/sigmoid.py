import numpy as np
from tinytorch.error import ForwardNotCalledError
from tinytorch.layers.module import Module
from tinytorch.parameter import Parameter
from typing import Sequence


class Sigmoid(Module):
  _sigma: np.ndarray | None

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    Args:
      x: shape (B, d)

    Output:
      y = exp(x)/(1+exp(x)) = 1 / (1 + exp(-x))
    """
    self._sigma = 1 / (1 + np.exp(-x))
    return self._sigma

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    """
    Args:
      grad_out: shape (B, d)

    Output:
      grad_in: shape (B, d), grad_out * s'(x)
    """
    if self._sigma is None:
      raise ForwardNotCalledError()

    # s'(x) = s(x)(1-s(x))
    return grad_out * self._sigma * (1 - self._sigma)

  def parameters(self) -> Sequence[Parameter]:
    return []
