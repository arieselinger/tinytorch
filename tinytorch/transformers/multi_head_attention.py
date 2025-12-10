from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import OneInputModule
from tinytorch.parameter import Parameter, create_he_normal_params
from tinytorch.transformers.kv_cache import KVCache
from tinytorch.transformers.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(OneInputModule):
  def __init__(self, num_heads: int, d_model: int, is_causal: bool) -> None:
    """
    Args:
      num_heads: number of attention heads
      d_model: model dimension
      is_causal: whether to apply causal masking in the attention scores
      key_padding_mask: mask to ignore certain keys (optional)
                        shape (..., Tq) instead of (..., T) since the cached keys are already
                        stored after masking
    """
    self._num_heads = num_heads
    self._d_model = d_model

    if d_model % num_heads != 0:
      raise ValueError("d_model should be a multiple of num_heads")

    self._Wq = create_he_normal_params(d_model, d_model)
    self._Wk = create_he_normal_params(d_model, d_model)
    self._Wv = create_he_normal_params(d_model, d_model)
    self._Wo = create_he_normal_params(d_model, d_model)

    self._attention_layer = ScaledDotProductAttention(is_causal)

    self._x: np.ndarray | None = None

  def parameters(self) -> Sequence[Parameter]:
    return [
      self._Wq,
      self._Wk,
      self._Wv,
      self._Wo,
      *self._attention_layer.parameters(),
    ]

  def forward(
    self,
    x: np.ndarray,
    cache: KVCache | None = None,
    key_padding_mask: np.ndarray | None = None,
  ) -> np.ndarray:
    """
    Args:
      x:  shape (..., Tq, d_model) (Tq = T if no kv-cache)
    """

    self._x = x

    Q = x @ self._Wq.data  # (..., Tq, d_model)

    # Compute keys and values for new provided tokens (Tq)
    K_new = x @ self._Wk.data  # (..., Tq, d_model)
    V_new = x @ self._Wv.data  # (..., Tq, d_model)

    # Mask padding keys if provided
    if key_padding_mask is not None:
      mask = key_padding_mask  # (..., Tq)
      mask = np.expand_dims(mask, axis=-1)  # (..., Tq, 1)
      K_new = K_new + (1 - mask) * -np.inf  # (..., Tq, d_model)

    # Get KV from cache if provided
    K, V = cache.append(self, K_new, V_new) if cache else (K_new, V_new)  # (..., T, d_model)

    # Shapes
    *leading, Tq, _ = Q.shape  # Tq is number of new queries (for training Tq = T)
    *_, T, _ = K.shape  # T is seq_len
    H = self._num_heads
    d_model = self._d_model
    d_head = d_model // H

    # Reshape into (..., H, T, d_head)
    Q_exp = Q.reshape(*leading, Tq, H, d_head).swapaxes(-2, -3)  # (..., H, Tq, d_head)
    K_exp = K.reshape(*leading, T, H, d_head).swapaxes(-2, -3)  # (..., H, T, d_head)
    V_exp = V.reshape(*leading, T, H, d_head).swapaxes(-2, -3)  # (..., H, T, d_head)

    # Attention and reshape back
    output = self._attention_layer(Q_exp, K_exp, V_exp)  # (..., H, Tq, d_head)
    output = output.swapaxes(-3, -2)  # (..., Tq, H, d_head)
    output = output.reshape(*leading, Tq, d_model)

    self._attention = output  # (..., Tq, d_model)
    return output @ self._Wo.data

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    """
    Args:
      grad_out: shape (..., Tq, d_model)
    """
    if self._x is None:
      raise ForwardNotCalledError()

    *leading, Tq, _ = grad_out.shape  # leading dims, num queries, d_model
    H = self._num_heads  # Number of heads
    d_model = self._d_model  # Model dimension
    d_head = d_model // H  # Dimension per head

    grad_out_2d = grad_out.reshape(-1, d_model)
    attention_2d = self._attention.reshape(-1, d_model)

    self._Wo.grad += attention_2d.T @ grad_out_2d  # (d_model, d_model)

    grad_out = grad_out @ self._Wo.data.T  # (..., Tq, d_model)
    grad_out = grad_out.reshape(*leading, Tq, H, d_head)
    grad_out = grad_out.swapaxes(-2, -3)  # (..., H, Tq, d_head)

    # (..., H, Tq, d), (..., H, T, d), (..., H, T, d)
    grad_Q_exp, grad_K_exp, grad_V_exp = self._attention_layer.backward(grad_out)

    T = grad_K_exp.shape[-2]
    grad_Q = grad_Q_exp.swapaxes(-3, -2).reshape(*leading, Tq, d_model)
    grad_K = grad_K_exp.swapaxes(-3, -2).reshape(*leading, T, d_model)
    grad_V = grad_V_exp.swapaxes(-3, -2).reshape(*leading, T, d_model)

    # K = [K_cache; K_new], V = [V_cache; V_new] and cached matrices are constant wrt x
    # Only backprop through the new Tq tokens (i.e. present in x)
    if Tq != T:
      grad_K = grad_K[..., -Tq:, :]  # (..., Tq, d_model)
      grad_V = grad_V[..., -Tq:, :]  # (..., Tq, d_model)

    grad_Q_2d = grad_Q.reshape(-1, d_model)  # (*, d_model)
    grad_K_2d = grad_K.reshape(-1, d_model)  # (*, d_model)
    grad_V_2d = grad_V.reshape(-1, d_model)  # (*, d_model)

    x_2d = self._x.reshape(-1, d_model)  # (*, d_model)

    self._Wq.grad += x_2d.T @ grad_Q_2d  # (d_model, d_model)
    self._Wk.grad += x_2d.T @ grad_K_2d  # (d_model, d_model)
    self._Wv.grad += x_2d.T @ grad_V_2d  # (d_model, d_model)

    # Propagate gradient back to input
    grad_in = grad_Q @ self._Wq.data.T  # (..., Tq, d_model)
    grad_in += grad_K @ self._Wk.data.T  # (..., Tq, d_model)
    grad_in += grad_V @ self._Wv.data.T  # (..., Tq, d_model)

    return grad_in
