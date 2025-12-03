from typing import Sequence
import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import OneInputModule
from tinytorch.parameter import Parameter
from tinytorch.training import TrainingContext


class Dropout(OneInputModule):
  def __init__(self, probability: float, context: TrainingContext) -> None:
    """
    Args:
        probability: Probability of dropping a unit (dropout rate)
        context: Training context to check training/eval mode.
    """
    if probability < 0.0 or probability >= 1.0:
      raise ValueError("Dropout probability must be in the range [0.0, 1.0).")
    self._p = probability
    self._context = context
    self._mask: np.ndarray | None = None

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    For each provided activation, we compute an *independent* mask ~ Bernouilli(q) with
    probability q = 1-p.

    For each neuron, the expectation is:
    E[mask * x] = qx => E[(mask * x) / q] = x

    So we compute y := (mask * x) / q during training.

    """
    if not self._context.is_training() or self._p == 0.0:
      return x

    keep_probability = 1.0 - self._p
    self._mask = (np.random.rand(*x.shape) < keep_probability).astype(x.dtype)
    return x * self._mask / keep_probability

  def backward(self, grad_out: np.ndarray) -> np.ndarray:
    if not self._context.is_training():
      raise RuntimeError("Backward should only be called on training mode")

    if self._mask is None:
      raise ForwardNotCalledError()

    keep_probability = 1.0 - self._p
    return grad_out * self._mask / keep_probability

  def parameters(self) -> Sequence[Parameter]:
    return []
