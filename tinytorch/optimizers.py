from typing import Sequence
from tinytorch.parameter import Parameter


class SGD:
  def __init__(
    self,
    parameters: Sequence[Parameter],
    learning_rate: float,
    weight_decay: float = 0.0,
  ):
    """
    Stochastic Gradient Descent

    With an optional L2 penalty term to the cost function (=> make the weights smaller)
    J(W) <- L(W) + weight_decay/2 * <W,W>

    dL/dW <- dL/W + weight_decay * W
    """
    self._parameters = parameters
    self._learning_rate = learning_rate
    self._weight_decay = weight_decay

  def step(self):
    for param in set(self._parameters):
      grad = param.grad + self._weight_decay * param.data
      param.data -= self._learning_rate * grad
