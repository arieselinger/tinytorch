from typing import Sequence

import numpy as np

from tinytorch.exceptions import ForwardNotCalledError
from tinytorch.module import CriterionModule
from tinytorch.parameter import Parameter


class SoftmaxCrossEntropyLoss(CriterionModule):
  def __init__(self) -> None:
    self._log_probs: np.ndarray | None = None
    self._shape: tuple[int, ...] | None = None
    self._num_points: int | None = None
    self._coordinates: tuple[np.ndarray, np.ndarray] | None = None
    self._flatten_shape: tuple[int, int] | None = None
    self._keep_index_flat: np.ndarray | None = None

  def forward(
    self,
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: np.ndarray | None = None,
  ) -> np.ndarray:
    """Softmax cross-entropy: L = -log(softmax(logits)[target])
    1. For each position:
      * Softmax
      * negative log likelihood
    2. Loss is the average over each position

    Args:
      logits: shape (..., n_classes)
      targets: shape (..., )
      ignore_index: positions to ignore in the loss and gradient (optional): shape (..., )

    Output:
      loss: mean negative log-likelihood
    """

    # Store this to reshape at the end of backward
    self._shape = logits.shape

    # Now for now on we only work with flattened versions of shape (N, n_classes)
    logits_flat = logits.reshape(-1, logits.shape[-1])  # shape (N, n_classes)
    targets_flat = targets.reshape(-1)  # shape (N, )
    self._flatten_shape = logits_flat.shape

    # Keep points that are not ignored
    if ignore_index is not None:
      keep_index_flat = ~ignore_index.reshape(-1).astype(bool)  # shape (N, )
      self._keep_index_flat = keep_index_flat
      logits_flat = logits_flat[keep_index_flat, :]
      targets_flat = targets_flat[keep_index_flat]

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
      or self._flatten_shape is None
    ):
      raise ForwardNotCalledError()

    # Compute softmax from cached log-probabilities
    softmax = np.exp(self._log_probs)

    # Create one-hot encoding of targets
    one_hot = np.zeros_like(softmax)
    one_hot[self._coordinates] = 1.0

    # Gradient: (softmax - one_hot) / num_points
    # Because of the mean reduction, each position contributes to 1/N to grdient)

    grad_in_flatten = grad_out * ((softmax - one_hot) / self._num_points)
    if self._keep_index_flat is not None:
      grad_in = np.zeros(self._flatten_shape)
      grad_in[self._keep_index_flat] = grad_in_flatten
    else:
      grad_in = grad_in_flatten
    grad_in = grad_in.reshape(*self._shape)

    return grad_in

  def parameters(self) -> Sequence[Parameter]:
    return []
