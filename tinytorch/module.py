from abc import ABCMeta, abstractmethod
from typing import Generic, ParamSpec, Sequence

import numpy as np
from tinytorch.parameter import Parameter

T = ParamSpec("T")


class Module(Generic[T], metaclass=ABCMeta):
  @abstractmethod
  def forward(self, *args: T.args, **kwargs: T.kwargs) -> np.ndarray: ...

  def __call__(self, *args: T.args, **kwargs: T.kwargs) -> np.ndarray:
    return self.forward(*args, **kwargs)

  @abstractmethod
  def backward(self, grad_out: np.ndarray) -> np.ndarray: ...

  @abstractmethod
  def parameters(self) -> Sequence[Parameter]: ...

  def zero_grad(self) -> None:
    for param in self.parameters():
      param.zero_grad()


SingleInputModule = Module[[np.ndarray]]
CriterionModule = Module[[np.ndarray, np.ndarray]]
