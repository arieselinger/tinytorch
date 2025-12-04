from typing import Sequence

import numpy as np

from tinytorch.module import TwoInputModule
from tinytorch.parameter import Parameter


class Add(TwoInputModule):
  def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Args:
      x1: shape (...)
      x2: shape (...)

    Output:
      y = x1 + x2, shape (...)
    """
    return x1 + x2

  def backward(self, grad_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
      grad_out: shape (...)

    Output:
      grad_in1: shape (...), equal to grad_out
      grad_in2: shape (...), equal to grad_out
    """
    return grad_out, grad_out

  def parameters(self) -> Sequence[Parameter]:
    return []
