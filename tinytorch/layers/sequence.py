import numpy as np
from typing import Sequence
from tinytorch.module import SingleInputModule
from tinytorch.parameter import Parameter


class Sequential(SingleInputModule):
  def __init__(self, layers: Sequence[SingleInputModule]) -> None:
    self.layers = layers

  def forward(self, x: np.ndarray) -> np.ndarray:
    for m in self.layers:
      x = m.forward(x)
    return x

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    grad = grad_out
    for m in reversed(self.layers):
      grad = m.backward(grad)
    return grad

  def parameters(self) -> Sequence[Parameter]:
    params: list[Parameter] = []
    for m in self.layers:
      params += m.parameters()
    return params
