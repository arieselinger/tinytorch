import numpy as np
from tinytorch.activations.relu import ReLU
from tinytorch.activations.sigmoid import Sigmoid
from tinytorch.activations.softmax import Softmax
from tinytorch.layers.linear import Linear
from tinytorch.layers.sequence import Sequential
from tinytorch.mse import MSELoss

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
    ReLU(),
    Linear(d2, d3),
    Sigmoid(),
    Linear(d3, d4),
    Softmax(),
  ]
)

# Forward
x = np.random.randn(batch_size, d1)
y_pred = model.forward(x)

# Compute loss
y_true = np.zeros_like(y_pred)
criterion = MSELoss()
loss = criterion.forward(y_pred, y_true)

# Backward
grad_loss = criterion.backward()
grad_x = model.backward(grad_loss)

model.zero_grad()
