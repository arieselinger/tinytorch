from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import OneInputModule
from tinytorch.parameter import Parameter, create_ones_params, create_zeros_params

EPSILON = 1e-6


class LayerNorm(OneInputModule):
  def __init__(self, d_in: int):
    """
    LayerNorm

    Args:
      d_in: dimension of features, vectors in forward should be of shape (..., d_in)

    Operations:
      1. Compute an independent point-wise normalization (0-mean and 1-std, to each feature vector)
      2. Same learned scale and shift parameters (one per feature dim) applied to each position

    Parameters:
      gamma: scaling factors, shape (d_in, )
      beta: scaling factors, shape (d_in, )
    """

    # Learned scaling factors
    self._gamma = create_ones_params(d_in)

    # Learning shifting factors
    self._beta = create_zeros_params(d_in)

    self._x_norm: np.ndarray | None = None
    self._sigma: np.ndarray | None = None

  def parameters(self) -> Sequence[Parameter]:
    return [self._gamma, self._beta]

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    Args:
      x: shape (..., d_in)
    """

    mu = np.mean(x, axis=-1, keepdims=True)  # shape (..., 1)
    sigma = np.std(x, axis=-1, keepdims=True)  # shape (..., 1)
    x_norm = (x - mu) / (sigma + EPSILON)  # shape (..., d_in)

    self._sigma = sigma
    self._x_norm = x_norm

    return x_norm * self._gamma.data + self._beta.data  # (..., d_in) * (d_in, ) + (d_in, )

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    """
    Args:
      grad_out: shape (..., d_in)


    Jacobian of x_norm wrt x:
    > J_ij = d_x_norm_i / d_x_j = 1/sigma (d_ij - 1/N - x_norm_i * x_norm_j / N)

    Gradient L wrt x_norm
    > dL / d_norm_i = g_i - mean_j(g_j) - x_norm_i * mean_j(x_norm_j * g_j)
    """

    if self._x_norm is None or self._sigma is None:
      raise ForwardNotCalledError()

    grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1])  # (B, d_in)
    x_norm_2d = self._x_norm.reshape(-1, self._x_norm.shape[-1])  # (B, d_in)

    self._beta.grad += np.sum(grad_out_2d, axis=0)  # shape (d_in, )
    self._gamma.grad += np.sum(grad_out_2d * x_norm_2d, axis=0)  # shape (d_in, )

    # Compute grad_x_norm (g = dL/dx_norm)
    g = grad_out * self._gamma.data  # (..., d_in) = (..., d_in) * (d_in,)

    # Compute grad_x
    mean_g = np.mean(g, axis=-1, keepdims=True)  # (..., 1)
    weighted_mean_g = np.mean(g * self._x_norm, axis=-1, keepdims=True)  # (..., 1)
    grad_x = (g - mean_g - weighted_mean_g * self._x_norm) / (self._sigma + EPSILON)  # (..., d_in)

    return grad_x
