from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import CriterionModule
from tinytorch.parameter import Parameter


class SoftmaxCrossEntropyLoss(CriterionModule):
  _log_probs: np.ndarray | None
  _shape: tuple[int, ...] | None
  _num_points: int | None
  _coordinates: tuple[np.ndarray[tuple[int]], np.ndarray[tuple[int]]] | None

  def forward(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Softmax cross-entropy: L = -log(softmax(logits)[target])
    1. For each position:
      * Softmax
      * negative log likelihood
    2. Loss is the average over each position

    Args:
      logits: shape (..., n_classes)
      targets: shape (..., )

    Output:
      loss: mean negative log-likelihood
    """

    self._shape = logits.shape

    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    num_points = targets_flat.shape[0]
    self._num_points = num_points

    # Numerically stable log-softmax (max subtraction for stability)
    logits_max = logits_flat.max(axis=-1, keepdims=True)
    logits_shifted = logits_flat - logits_max
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))
    log_probs = logits_shifted - log_sum_exp
    self._log_probs = log_probs

    # Extract target class log-probs for each position
    rows = np.arange(num_points)
    cols = targets_flat
    self._coordinates = rows, cols
    target_log_probs = log_probs[rows, cols]

    return -target_log_probs.mean()

  def backward(self, grad_out: np.ndarray = np.array(1.0)) -> np.ndarray:
    """Gradient: [softmax(logits) - one_hot(targets)] / num_points

    Output:
      grad_in: shape (..., n_classes)
    """
    if (
      self._log_probs is None
      or self._shape is None
      or self._coordinates is None
      or self._num_points is None
    ):
      raise ForwardNotCalledError()

    # Compute softmax from cached log-probabilities
    softmax = np.exp(self._log_probs)

    # Create one-hot encoding of targets
    one_hot = np.zeros_like(softmax)
    one_hot[self._coordinates] = 1.0

    # Gradient: (softmax - one_hot) / num_points
    # Because of the mean reduction, each position contributes to 1/N to grdient)
    grad_in = grad_out * ((softmax - one_hot) / self._num_points)
    return grad_in.reshape(*self._shape)

  def parameters(self) -> Sequence[Parameter]:
    return []
