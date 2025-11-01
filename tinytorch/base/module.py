from abc import ABCMeta, abstractmethod
from typing import Any, Sequence

from tinytorch.base.parameter import Parameter


class Module(metaclass=ABCMeta):
  @abstractmethod
  def forward(self, *args: Any, **kwargs: Any) -> Any: ...

  @abstractmethod
  def backward(self, *args: Any, **kwargs: Any) -> Any: ...

  @abstractmethod
  def parameters(self) -> Sequence[Parameter]: ...

  def zero_grad(self) -> None:
    for param in self.parameters():
      param.zero_grad()
