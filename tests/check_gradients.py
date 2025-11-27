import numpy as np

from tinytorch.layers.linear import Linear
from tinytorch.module import OneInputModule

EPSILON = 1e-5
TOLERANCE = 1e-4


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

  grad_in = np.zeros_like(arr)
  it = np.nditer(arr, flags=["multi_index"], op_flags=[["readwrite"]])

  while not it.finished:
    idx = it.multi_index
    original_value = arr[idx]

    arr[idx] = original_value + EPSILON
    loss_pos = module.forward(x).sum()

    arr[idx] = original_value - EPSILON
    loss_neg = module.forward(x).sum()

    grad_in[idx] = (loss_pos - loss_neg) / (2 * EPSILON)
    arr[idx] = original_value
    it.iternext()

  return grad_in


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


if __name__ == "__main__":
  module = Linear(10, 100)
  x = np.random.randn(64, 10)
  print(compare_gradients(module, x))
