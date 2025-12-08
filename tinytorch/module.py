from abc import ABCMeta, abstractmethod
from typing import Generic, ParamSpec, Sequence, TypeVar

import numpy as np

from tinytorch.parameter import Parameter

In = ParamSpec("In")
Out = TypeVar("Out")
GradIn = TypeVar("GradIn")


class _Module(Generic[In, Out, GradIn], metaclass=ABCMeta):
  @abstractmethod
  def forward(self, *args: In.args, **kwargs: In.kwargs) -> Out: ...

  def __call__(self, *args: In.args, **kwargs: In.kwargs) -> Out:
    return self.forward(*args, **kwargs)

  @abstractmethod
  def backward(self, grad_out: Out) -> GradIn: ...

  @abstractmethod
  def parameters(self) -> Sequence[Parameter]: ...

  def zero_grad(self) -> None:
    for param in self.parameters():
      param.zero_grad()


# Classes to inherit from
OneInputModule = _Module[[np.ndarray], np.ndarray, np.ndarray]
TwoInputModule = _Module[[np.ndarray, np.ndarray], np.ndarray, tuple[np.ndarray, np.ndarray]]
ThreeInputModule = _Module[
  [np.ndarray, np.ndarray, np.ndarray], np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]
]
CriterionModule = _Module[[np.ndarray, np.ndarray], np.ndarray, np.ndarray]
OneInputModuleNoGrad = _Module[[np.ndarray], np.ndarray, None]
