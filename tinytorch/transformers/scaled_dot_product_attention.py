from typing import Sequence

import numpy as np

from tinytorch.activations.softmax import Softmax
from tinytorch.layers.matmul import MatMul
from tinytorch.module import ThreeInputModule
from tinytorch.parameter import Parameter


class ScaledDotProductAttention(ThreeInputModule):
  def __init__(self, causal_mask: bool) -> None:
    """
    ScaledDotProductAttention

    Args:
      causal_mask: if true, apply causal mask so that each query can only attend its previous keys.
                   The mask aligns to the bottom-right, so the last query attends to all keys,
                   second-to-last attends to all but the last key, etc.
                    [[0 0 -inf -inf -inf]
                     [0 0    0 -inf -inf]
                     [0 0    0    0 -inf] <- second to last query
                     [0 0    0    0   0]] <- last query
    """
    self._matmul1 = MatMul()
    self._softmax = Softmax()
    self._matmul2 = MatMul()
    self._causal_mask = causal_mask

  def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:  # noqa: N803
    num_queries = Q.shape[-2]
    num_keys = K.shape[-2]

    mask = None
    if self._causal_mask and num_queries >= 1:
      mask = np.triu(-np.inf * np.ones((num_queries, num_keys)), k=num_keys - num_queries + 1)

    scale = 1 / np.sqrt(K.shape[-1])
    self._scale: np.floating = scale

    scores = self._matmul1(Q, K.swapaxes(-1, -2))
    scores = scores * scale
    if mask is not None:
      scores = scores + mask
    attention = self._softmax(scores)
    output = self._matmul2(attention, V)
    return output

  def backward(self, grad_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g, grad_V = self._matmul2.backward(grad_out)  # noqa: N806
    g = self._softmax.backward(g)
    g = g * self._scale
    grad_Q, grad_K_transpose = self._matmul1.backward(g)  # noqa: N806
    grad_K = grad_K_transpose.swapaxes(-1, -2)  # noqa: N806
    return grad_Q, grad_K, grad_V

  def parameters(self) -> Sequence[Parameter]:
    return []
