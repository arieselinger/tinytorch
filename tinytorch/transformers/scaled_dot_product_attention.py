from typing import Sequence

import numpy as np

from tinytorch.activations.softmax import Softmax
from tinytorch.layers.matmul import MatMul
from tinytorch.module import ThreeInputModule
from tinytorch.parameter import Parameter


class ScaledDotProductAttention(ThreeInputModule):
  def __init__(self, d_k: int) -> None:
    """
    Docstring for __init__

    Args:
      d_k: dim of keys and queries
    """
    self._matmul1 = MatMul()
    self._softmax = Softmax()
    self._scale: np.floating = 1 / np.sqrt(d_k)
    self._matmul2 = MatMul()

  def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:  # noqa: N803
    seq_len = Q.shape[-2]
    mask = np.triu(-np.inf * np.ones((seq_len, seq_len)), k=1)

    scores = self._matmul1(Q, K.swapaxes(-1, -2))
    scores = scores * self._scale
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
