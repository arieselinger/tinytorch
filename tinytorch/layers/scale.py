from typing import Sequence

import numpy as np

from tinytorch.module import OneInputModule
from tinytorch.parameter import Parameter


class Scale(OneInputModule):
  def __init__(self, scale: np.floating):
    self._scale = scale

  def forward(self, x: np.ndarray) -> np.ndarray:
    return self._scale * x

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    return self._scale * grad_out

  def parameters(self) -> Sequence[Parameter]:
    return []
