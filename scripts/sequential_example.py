import numpy as np
from tinytorch.linear import Linear
from tinytorch.sequence import Sequential

# Constants
batch_size = 12
d1 = 10
d2 = 5
d3 = 4
d4 = 17

# Define sequential model
model = Sequential(
  [
    Linear(d1, d2),
    Linear(d2, d3),
    Linear(d3, d4),
  ]
)

# Forward
x = np.random.randn(batch_size, d1)
y = model(x)

# Backward
grad_y = np.random.randn(batch_size, d4)
grad_x = model.backward(grad_y)

model.zero_grad()
