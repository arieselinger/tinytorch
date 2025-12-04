from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import OneInputModule
from tinytorch.parameter import Parameter, create_he_normal_params, create_zeros_params


class Linear(OneInputModule):
  def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
    """
    Args:
      d_in: input size
      d_out: output size

    Parameters:
      W: shape (d_in, d_out)
      b: shape (d_out, )
    """

    # Trainable weights
    self.W = create_he_normal_params(d_in, d_out)  # He init: great for ReLU
    self.b = create_zeros_params(d_out) if bias else None

    # Cached for backward pass
    self._x: np.ndarray | None = None

  def parameters(self) -> Sequence[Parameter]:
    if self.b is None:
      return [self.W]
    return [self.W, self.b]

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    The same linear transform is applied "point-wise" i.e. independently to each feature vector.
    We consider the last axis as the feature dimension.

    Args:
      x: shape (..., d_in) (... means arbitrary batch dimensions)

    Output:
      y = Wx + b: shape (..., d_out)

    """
    self._x = x

    # NumPy broadcasting rules apply for adding b to each output feature vector
    if self.b is None:
      return x @ self.W.data
    return x @ self.W.data + self.b.data  # (..., d_in) @ (d_in, d_out) + (d_out,) -> (..., d_out)

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    """
    Args:
      grad_out: gradient of loss wrt to output, (..., d_out)

    Output:
      grad_in: gradient of loss wrt to input, (..., d_in)

    Updates:
      W.grad: shape (d_in, d_out)
      b.grad: shape (d_out, )

    Gradient derivation for b:

    First we consider the 2-D case:

    Vector b (dim_out, ) is broadcasted to match xW (B, dim_out) in
    y = xW + b

    Equivalent to: b_expanded = np.ones((batch_size, 1)) @ b

    Which helps us derive the gradient:

    grad_b[loss(np.ones((batch_size, 1)) @ b)]
    = np.ones((batch_size, 1)).T @ grad_out
    = np.ones((1, batch_size)) @ grad_out
    = grad_out.sum(axis=0, keepdims=False)
    => "sum sample gradients over the batch"

    Note: the gradient wrt W is also summing over each batch sample. grad_W = x.T @ grad_out
          which sums matrices sum(x_i^T @ grad_out_i) for each sample i

    """

    if self._x is None:
      raise ForwardNotCalledError()

    grad_in = grad_out @ self.W.data.T

    # For 2D tensors we want to return the following:
    #   self.W.grad += self._x.T @ grad_out
    #   self.b.grad += grad_out.sum(axis=0)
    #
    # Where: self._x.T @ grad_out has shape (d_in, B) @ (B, d_out) = (d_in, d_out)
    #
    # This sums over the batch dimension, accumulating gradients from all samples
    # => Makes sense since it's the same linear layer applied point-wise (for each feature vector)
    # => For higher-dimensional tensors, flatten all batch dimensions first to get the same effect
    x_flatten = self._x.reshape(-1, self._x.shape[-1])
    grad_out_flatten = grad_out.reshape(-1, grad_out.shape[-1])
    self.W.grad += x_flatten.T @ grad_out_flatten

    if self.b is not None:
      self.b.grad += grad_out_flatten.sum(axis=0)

    # An other approach instead of flattening is using the Einstein notation
    # to sum over all batches dimensions:
    # self.W.grad += np.einsum('...i,...j=>ij', self._x, grad_out)
    # self.b.grad += np.einsum('...j->j', grad_out)
    # This might be much slower computationally.

    return grad_in
