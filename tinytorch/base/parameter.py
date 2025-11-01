import numpy as np


class Parameter:
  def __init__(self, data: np.ndarray) -> None:
    self.data = data
    self.grad = np.zeros_like(data)

  def __repr__(self) -> str:
    return f"<Parameter data={self.data} grad={self.grad} shape={self.data}>"

  def zero_grad(self) -> None:
    self.grad = np.zeros_like(self.data)


def he_normal_params(*shape: int) -> Parameter:
  d_in = shape[0]
  std = np.sqrt(2.0 / d_in)
  return Parameter(np.random.randn(*shape) * std)


def zeros_params(*shape: int) -> Parameter:
  return Parameter(np.zeros(shape))
