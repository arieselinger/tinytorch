from typing import Sequence

import numpy as np

from tinytorch.module import OneInputModuleNoGrad
from tinytorch.parameter import Parameter


class PositionalEncoding(OneInputModuleNoGrad):
  def __init__(self, max_seq_len: int, d_model: int):
    self._max_seq_len = max_seq_len
    self._d_model = d_model
    self._positional_encoding = self._create_sinusoidal_encoding()

  def _create_sinusoidal_encoding(self) -> np.ndarray:
    """
    Generate a vector of shape (max_seq_len, d_model)
    """

    d_model = self._d_model
    features = np.arange(d_model)  # 2i -> 2i ; 2i+1 -> 2i+1
    features -= features % 2  # 2i -> 2i ; 2i+1 -> 2i
    features = 1 / (1e4 ** (features / d_model))

    features = features.reshape(1, -1)  # (1, d_model)
    positions = np.arange(self._max_seq_len).reshape(-1, 1)  # (num_tokens, 1)

    embeddings = positions @ features  # (num_tokens, d_model)
    embeddings[:, ::2] = np.sin(embeddings[:, ::2])
    embeddings[:, 1::2] = np.cos(embeddings[:, 1::2])

    return embeddings

  def forward(self, x: np.ndarray) -> np.ndarray:
    """Add the positional encoding to x"""
    num_tokens = x.shape[-2]
    return x + self._positional_encoding[:num_tokens]

  def backward(self, grad_out: np.ndarray) -> None:
    return None

  def parameters(self) -> Sequence[Parameter]:
    return []
