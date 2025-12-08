from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import OneInputModuleNoGrad
from tinytorch.parameter import Parameter, create_normal_params


class Embedding(OneInputModuleNoGrad):
  def __init__(self, vocab_size: int, d_model: int) -> None:
    """
    Args:
      vocab_size: size of the vocabulary
      d_model: dimension of the embeddings
    """
    self._vocab_size = vocab_size
    self._d_model = d_model

    self._tokens: np.ndarray | None = None

    self.weights = create_normal_params(vocab_size, d_model, std=0.02)

  def parameters(self) -> Sequence[Parameter]:
    return [self.weights]

  def forward(self, tokens: np.ndarray) -> np.ndarray:
    """
    Args:
      tokens: shape (...), integer indices in [0, vocab_size)

    Output:
      y: shape (..., d_model), embeddings corresponding to the input indices
    """
    self._tokens = tokens
    return self.weights.data[tokens]

  def backward(self, grad_out: np.ndarray) -> None:
    if self._tokens is None:
      raise ForwardNotCalledError()

    # The following does not work when there are repeated tokens in self._tokens
    # This would show the right array but only update the first occurrence in the gradient
    # self._W.grad[self._tokens] += grad_out
    # Instead we need this:
    np.add.at(self.weights.grad, self._tokens, grad_out)
