from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from tinytorch.transformers.multi_head_attention import MultiHeadAttention


class KVCache:
  def __init__(self):
    self._K: dict[MultiHeadAttention, np.ndarray] = {}
    self._V: dict[MultiHeadAttention, np.ndarray] = {}

  def reset(self, module: MultiHeadAttention) -> None:
    self._K.pop(module, None)
    self._V.pop(module, None)

  def get(self, module: MultiHeadAttention) -> tuple[np.ndarray | None, np.ndarray | None]:
    return self._K.get(module), self._V.get(module)

  def append(
    self,
    module: MultiHeadAttention,
    K_new: np.ndarray,  # noqa: N803
    V_new: np.ndarray,  # noqa: N803
  ) -> tuple[np.ndarray, np.ndarray]:
    K_cached = self._K.get(module)
    V_cached = self._V.get(module)

    # Concatenate along time axis:
    # (B, T_cached, d_model) + (B, T_q, d_model) -> (B, T_cached + T_q, d_model)
    self._K[module] = np.concatenate([K_cached, K_new], axis=1) if K_cached is not None else K_new
    self._V[module] = np.concatenate([V_cached, V_new], axis=1) if V_cached is not None else V_new
    return self._K[module], self._V[module]

  def reset_all(self) -> None:
    self._K.clear()
    self._V.clear()
