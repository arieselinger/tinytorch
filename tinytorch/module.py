from abc import ABCMeta, abstractmethod
from typing import Any, Sequence

from tinytorch.parameter import Parameter


class Module(metaclass=ABCMeta):
  @abstractmethod
  def forward(self, *args: Any, **kwargs: Any) -> Any: ...

  @abstractmethod
  def backward(self, *args: Any, **kwargs: Any) -> Any: ...

  @abstractmethod
  def parameters(self) -> Sequence[Parameter]: ...

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self.forward(*args, **kwargs)

  def zero_grad(self) -> None:
    for param in self.parameters():
      param.zero_grad()
