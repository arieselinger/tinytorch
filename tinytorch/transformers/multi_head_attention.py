from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import OneInputModule
from tinytorch.parameter import Parameter, create_he_normal_params
from tinytorch.transformers.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(OneInputModule):
  def __init__(self, num_heads: int, d_model: int, causal_mask: bool = True):
    self._num_heads = num_heads
    self._d_model = d_model

    if d_model % num_heads != 0:
      raise ValueError("d_model should be a multiple of num_heads")

    self._Wq = create_he_normal_params(d_model, d_model)
    self._Wk = create_he_normal_params(d_model, d_model)
    self._Wv = create_he_normal_params(d_model, d_model)
    self._Wo = create_he_normal_params(d_model, d_model)

    self._attention_layer = ScaledDotProductAttention(causal_mask)

    self._x: np.ndarray | None = None

  def parameters(self) -> Sequence[Parameter]:
    return [
      self._Wq,
      self._Wk,
      self._Wv,
      self._Wo,
      *self._attention_layer.parameters(),
    ]

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    Args:
      x: shape (B, T_q, d_model) (T_q = T if no kv-cache)
    """

    self._x = x

    Q = x @ self._Wq.data  # (B, Tq, d_model)

    # TODO: ADD KV CACHE - Augment K and V with cached T - Tq rows
    K = x @ self._Wk.data  # (B, Tq, d_model) after KV cache it changes to (B, T, d_model)
    V = x @ self._Wv.data  # (B, Tq, d_model) after KV cache it changes to (B, T, d_model)

    # Shapes
    B, Tq, _ = Q.shape  # Tq is number of new queries (for training Tq = T)
    _, T, _ = K.shape  # T is seq_len
    H = self._num_heads
    d_model = self._d_model
    d_head = d_model // H

    # Reshape into (B, H, T, d_head)
    Q_exp = Q.reshape(B, Tq, H, d_head).transpose(0, 2, 1, 3)  # (B, H, Tq, d_head)
    K_exp = K.reshape(B, T, H, d_head).transpose(0, 2, 1, 3)  # (B, H, T, d_head)
    V_exp = V.reshape(B, T, H, d_head).transpose(0, 2, 1, 3)  # (B, H, T, d_head)

    # Reshape (B, Tq, H, d_head) and concat (B, Tq, d_model = H * d_head)
    output = self._attention_layer(Q_exp, K_exp, V_exp)  # (B, H, Tq, d_head)
    output = output.transpose(0, 2, 1, 3)  # (B, Tq, H, d_head)
    output = output.reshape(B, Tq, d_model)

    self._attention = output  # (B, Tq, d_model)
    return output @ self._Wo.data

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    """
    Args:
      grad_out: shape (B, Tq, d_model)
    """
    if self._x is None:
      raise ForwardNotCalledError()

    B, Tq, _ = grad_out.shape  # Batch size, num queries
    H = self._num_heads  # Number of heads
    d_model = self._d_model  # Model dimension
    d_head = d_model // H  # Dimension per head

    grad_out_2d = grad_out.reshape(-1, d_model)
    attention_2d = self._attention.reshape(-1, d_model)

    self._Wo.grad += attention_2d.T @ grad_out_2d  # (d_model, d_model)

    grad_out = grad_out @ self._Wo.data.T  #  (B, Tq, d_model)
    grad_out = grad_out.reshape(B, Tq, H, d_head)
    grad_out = grad_out.transpose(0, 2, 1, 3)  # (B, H, Tq, d_head)

    # (B, H, Tq, d), (B, H, T, d), (B, H, T, d)
    grad_Q_exp, grad_K_exp, grad_V_exp = self._attention_layer.backward(grad_out)

    _, _, T, _ = grad_K_exp.shape
    grad_Q = grad_Q_exp.transpose(0, 2, 1, 3).reshape(B, Tq, d_model)
    grad_K = grad_K_exp.transpose(0, 2, 1, 3).reshape(B, T, d_model)
    grad_V = grad_V_exp.transpose(0, 2, 1, 3).reshape(B, T, d_model)

    # K = [K_cache; K_new], V = [V_cache; V_new] and cached matrices are constant wrt x
    # Only backprop through the new Tq tokens (i.e. present in x)
    if Tq != T:
      grad_K = grad_K[:, -Tq:, :]  # (B, Tq, d_model)
      grad_V = grad_V[:, -Tq:, :]  # (B, Tq, d_model)

    grad_Q_2d = grad_Q.reshape(-1, d_model)  # (B*Tq, d_model)
    grad_K_2d = grad_K.reshape(-1, d_model)  # (B*Tq, d_model)
    grad_V_2d = grad_V.reshape(-1, d_model)  # (B*Tq, d_model)

    x_2d = self._x.reshape(-1, d_model)  # (B*Tq, d_model)

    self._Wq.grad += x_2d.T @ grad_Q_2d  # (d_model, d_model)
    self._Wk.grad += x_2d.T @ grad_K_2d  # (d_model, d_model)
    self._Wv.grad += x_2d.T @ grad_V_2d  # (d_model, d_model)

    # Propagate gradient back to input
    grad_in = grad_Q @ self._Wq.data.T  # (B, Tq, d_model)
    grad_in += grad_K @ self._Wk.data.T  # (B, Tq, d_model)
    grad_in += grad_V @ self._Wv.data.T  # (B, Tq, d_model)

    return grad_in
