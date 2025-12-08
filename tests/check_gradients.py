from typing import Any, Callable, Sequence

import numpy as np

from tinytorch.layers.linear import Linear
from tinytorch.module import CriterionModule, OneInputModule, ThreeInputModule

EPSILON = 1e-5
TOLERANCE = 1e-4


def _finite_differences_sum(
  forward_fn: Callable[[], np.ndarray], arrays: Sequence[np.ndarray]
) -> list[np.ndarray]:
  """Central finite differences of sum(forward_fn()) for each array in `arrays`."""
  grads: list[np.ndarray] = []
  for arr in arrays:
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"], op_flags=[["readwrite"]])
    while not it.finished:
      idx = it.multi_index
      original_value = arr[idx]

      arr[idx] = original_value + EPSILON
      loss_pos = np.asarray(forward_fn()).sum()

      arr[idx] = original_value - EPSILON
      loss_neg = np.asarray(forward_fn()).sum()

      grad[idx] = (loss_pos - loss_neg) / (2 * EPSILON)
      arr[idx] = original_value
      it.iternext()
    grads.append(grad)
  return grads


def _compute_numerical_gradients(
  module: OneInputModule,
  x: np.ndarray,
  arr: np.ndarray,
) -> np.ndarray:
  """
  Estimate the numerical gradients of sum(module(x)) w.r.t. arr using central finite differences.

  Args:
    module: Module to check
    x: Input to the forward pass
    arr: Array to compute gradients with respect to, could be `x` or `p.data()`

  Output:
    Numerical gradients of loss = sum(module.forward(x)) w.r.t. arr: or d(1^T * module(x)) / d_arr
  """
  return _finite_differences_sum(lambda: module.forward(x), [arr])[0]


def compare_gradients(
  module: OneInputModule,
  x: np.ndarray,
) -> bool:
  y_pred = module.forward(x)

  # dsum(y)/dy_i = 1
  grad_y = np.ones_like(y_pred)

  grad_x = module.backward(grad_y)
  num_grad_x = _compute_numerical_gradients(module, x, x)
  if not np.allclose(grad_x, num_grad_x, atol=TOLERANCE):
    return False

  for p in module.parameters():
    num_grad_p = _compute_numerical_gradients(module, x, p.data)
    if not np.allclose(p.grad, num_grad_p, atol=TOLERANCE):
      return False

  return True


def compare_three_input_gradients(
  module: ThreeInputModule,
  q: np.ndarray,
  k: np.ndarray,
  v: np.ndarray,
) -> bool:
  """Check gradients for modules taking three ndarray inputs (e.g., attention)."""
  out = module.forward(q, k, v)
  grad_out = np.ones_like(out)

  grad_q, grad_k, grad_v = module.backward(grad_out)

  num_grad_q, num_grad_k, num_grad_v = _finite_differences_sum(
    lambda: module.forward(q, k, v), [q, k, v]
  )

  return (
    np.allclose(grad_q, num_grad_q, atol=TOLERANCE)
    and np.allclose(grad_k, num_grad_k, atol=TOLERANCE)
    and np.allclose(grad_v, num_grad_v, atol=TOLERANCE)
  )


def compare_criterion_gradients(
  criterion: CriterionModule,
  logits: np.ndarray,
  targets: np.ndarray,
  **forward_kwargs: Any,
) -> bool:
  """
  Check gradients for a criterion module (loss function).

  Args:
    criterion: Criterion module to test
    logits: Input logits
    targets: Target values
    **forward_kwargs: Additional keyword arguments passed to forward

  Returns:
    True if analytical gradients match numerical gradients
  """
  # Forward pass
  criterion.forward(logits, targets, **forward_kwargs)

  # Backward pass (criterion returns gradient directly, grad_out defaults to 1.0)
  grad_logits = criterion.backward(np.array(1.0))

  # Numerical gradient
  num_grad_logits = _finite_differences_sum(
    lambda: criterion.forward(logits, targets, **forward_kwargs), [logits]
  )[0]

  return np.allclose(grad_logits, num_grad_logits, atol=TOLERANCE)


if __name__ == "__main__":
  module = Linear(10, 100)
  x = np.random.randn(64, 10)
  print(compare_gradients(module, x))
