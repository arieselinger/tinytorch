from typing import Sequence

import numpy as np

from tinytorch.activations.softmax import Softmax
from tinytorch.layers.matmul import MatMul
from tinytorch.module import ThreeInputModule
from tinytorch.parameter import Parameter


class ScaledDotProductAttention(ThreeInputModule):
  def __init__(self, is_causal: bool) -> None:
    """
    ScaledDotProductAttention

    Args:
      is_causal: Whether a causal mask should be applied.
                 Causal mask is applied to  attention scores before softmax so that each query can
                 only attend keys from previous time positions.
                 The mask aligns to the bottom-right, so the last query attends
                 to all keys, second-to-last attends to all but the last key, etc.
                 [[0 0 -inf -inf -inf]
                  [0 0    0 -inf -inf]
                  [0 0    0    0 -inf] <- second to last query
                  [0 0    0    0   0]] <- last query
                Note: padding mask should be applied outside to this module to padding keys.
    """
    self._matmul1 = MatMul()
    self._softmax = Softmax()
    self._matmul2 = MatMul()
    self._is_causal = is_causal

  def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:  # noqa: N803
    """
    Compute: softmax(Q@K^T / sqrt(d_k))@V

    Args:
      Q: Queries shape (..., Tq, d_k)
      K: Keys shape (..., T, d_k)
      V: Values shape (..., T, d_v)
    """

    Tq = Q.shape[-2]  # num queries
    *_, T, d_k = K.shape  # seq len (num keys/values)

    causal_mask = None
    if self._is_causal and Tq >= 1:
      causal_mask = np.triu(-np.inf * np.ones((Tq, T)), k=T - Tq + 1)

    scale = 1 / np.sqrt(d_k)
    self._scale: np.floating = scale

    scores = self._matmul1(Q, K.swapaxes(-1, -2))
    scores = scores * scale
    if causal_mask is not None:
      scores += causal_mask
    attention = self._softmax(scores)
    output = self._matmul2(attention, V)
    return output

  def backward(self, grad_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g, grad_V = self._matmul2.backward(grad_out)
    g = self._softmax.backward(g)
    g = g * self._scale
    grad_Q, grad_K_transpose = self._matmul1.backward(g)
    grad_K = grad_K_transpose.swapaxes(-1, -2)
    return grad_Q, grad_K, grad_V

  def parameters(self) -> Sequence[Parameter]:
    return []
