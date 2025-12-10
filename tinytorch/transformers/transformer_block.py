from typing import Sequence

import numpy as np

from tinytorch.activations.gelu import GELU
from tinytorch.layers.layer_norm import LayerNorm
from tinytorch.layers.linear import Linear
from tinytorch.module import OneInputModule
from tinytorch.parameter import Parameter
from tinytorch.transformers.kv_cache import KVCache
from tinytorch.transformers.multi_head_attention import MultiHeadAttention


class TransformerBlock(OneInputModule):
  def __init__(self, num_heads: int, d_model: int, d_ff: int, is_causal: bool):
    self._num_heads = num_heads
    self._d_model = d_model
    self._is_causal = is_causal

    # First block
    self._attention = MultiHeadAttention(num_heads, d_model, is_causal)
    self._layernorm1 = LayerNorm(d_model)

    # Second block
    self._feedforward1 = Linear(d_model, d_ff, bias=True)
    self._feedforward2 = Linear(d_ff, d_model, bias=True)
    self._layernorm2 = LayerNorm(d_model)
    self._gelu = GELU()

  def parameters(self) -> Sequence[Parameter]:
    return [
      *self._attention.parameters(),
      *self._layernorm1.parameters(),
      *self._layernorm2.parameters(),
      *self._feedforward1.parameters(),
      *self._feedforward2.parameters(),
    ]

  def forward(
    self,
    x: np.ndarray,
    cache: KVCache | None = None,
    key_padding_mask: np.ndarray | None = None,
  ) -> np.ndarray:
    """
    Compute the transfomer block using PRE-norm architectures (norm is applied before each sub-layer
    but not to residual connection)
    """
    # Attention: x -> x + mha(norm(x))
    h = self._layernorm1(x)
    h = self._attention.forward(h, cache=cache, key_padding_mask=key_padding_mask)
    x = h + x

    # feature-wise feed-forward: x -> x + linear(gelu(linear(norm(x))))
    h = self._layernorm2(x)
    h = self._feedforward1(h)
    h = self._gelu(h)
    h = self._feedforward2(h)
    x = h + x
    return x

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    grad_h = self._feedforward2.backward(grad_out)
    grad_h = self._gelu.backward(grad_h)
    grad_h = self._feedforward1.backward(grad_h)
    grad_x = grad_out + self._layernorm2.backward(grad_h)

    grad_h = self._attention.backward(grad_x)
    grad_x = grad_x + self._layernorm1.backward(grad_h)

    return grad_x
