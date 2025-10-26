import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Sequence

from tinytorch.parameter import Parameter


class Module(metaclass=ABCMeta):
  @abstractmethod
  def forward(self, x: np.ndarray) -> np.ndarray: ...

  @abstractmethod
  def backward(self, grad_out: np.ndarray) -> np.ndarray: ...

  @abstractmethod
  def parameters(self) -> Sequence[Parameter]: ...

  def __call__(self, x: np.ndarray) -> np.ndarray:
    return self.forward(x)

  def zero_grad(self) -> None:
    for param in self.parameters():
      param.zero_grad()
