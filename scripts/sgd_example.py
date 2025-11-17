import numpy as np
from tinytorch.activations.relu import ReLU
from tinytorch.activations.softmax import Softmax
from tinytorch.layers.linear import Linear
from tinytorch.layers.sequence import Sequential
from tinytorch.criteria.mse import MSELoss
from tinytorch.optimizers import SGD

# Constants
batch_size = 64
num_epochs = 5000
d1 = 10
d2 = 100
d3 = 100
d4 = 10
learning_rate = 0.01
weight_decay = 0.001

# Define model
model = Sequential(
  [
    Linear(d1, d2),
    ReLU(),
    Linear(d2, d3),
    ReLU(),
    Linear(d3, d4),
    Softmax(),
  ]
)

print("> Num parameters:", sum(p.data.size for p in model.parameters()))

# Define loss function
criterion = MSELoss()

# Data
x = np.random.randn(batch_size, d1)
y_true = np.random.randn(batch_size, d4)

# Optimizer
optimizer = SGD(model.parameters(), learning_rate, weight_decay)

# Training loop
for epoch in range(num_epochs):
  # Forward pass
  y_pred = model(x)
  loss = criterion(y_pred, y_true)

  # Backward pass
  grad_loss = criterion.backward()
  grad_x = model.backward(grad_loss)

  # Update parameters
  optimizer.step()

  if epoch % 100 == 0:
    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.6f}")

  model.zero_grad()
