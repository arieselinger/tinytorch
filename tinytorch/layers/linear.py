from typing import Sequence
import numpy as np
from tinytorch.error import ForwardNotCalledError
from tinytorch.module import Module
from tinytorch.parameter import Parameter, he_normal_params, zeros_params
from tinytorch.types import SingleInputModuleProtocol


class Linear(Module, SingleInputModuleProtocol):
  def __init__(self, d_in: int, d_out: int) -> None:
    """
    Args:
      d_in: input size
      d_out: output size

    Parameters:
      W: shape (d_in, d_out)
      b: shape (d_out, )
    """

    # Trainable weights
    self.W = he_normal_params(d_in, d_out)  # He init: great for ReLU
    self.b = zeros_params(d_out)

    # Non-trainable variables
    self._x: np.ndarray | None = None

  def parameters(self) -> Sequence[Parameter]:
    return [self.W, self.b]

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    Args:
      x: shape (B, d_in)

    Output:
      y = Wx + b: size (d_out, B)
    """
    self._x = x
    return x @ self.W.data + self.b.data  # (B, d_out) + (, d_out) -> (B, d_out)

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    """
    Args:
      grad_out: gradient of loss wrt to output, (B, d_out)

    Output:
      grad_in: gradient of loss wrt to input, (B, d_in)

    Updates:
      W.grad: shape (d_out, d_in)
      b.grad: shape (d_out, )

    --------
    Gradient wrt b

    Vector b (dim_out, ) is broadcasted to match xW (B, dim_out) in
    y = xW + b

    Equivalent to: b_expanded = np.ones((batch_size, 1)) @ b

    The former option gives us easily the solution for the gradient:

    grad_b[loss(np.ones((batch_size, 1)) @ b)]
    = np.ones((batch_size, 1)).T @ grad_out
    = np.ones((1, batch_size)) @ grad_out
    = grad_out.sum(axis=0, keepdims=False)

    => "sum sample gradients over the batch"
    """

    if self._x is None:
      raise ForwardNotCalledError()

    grad_in = grad_out @ self.W.data.T
    self.W.grad += self._x.T @ grad_out
    self.b.grad += grad_out.sum(axis=0)

    return grad_in
