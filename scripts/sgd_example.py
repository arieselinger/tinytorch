import numpy as np

from tinytorch.activations.relu import ReLU
from tinytorch.criteria.mse import MSELoss
from tinytorch.layers.dropout import Dropout
from tinytorch.layers.layer_norm import LayerNorm
from tinytorch.layers.linear import Linear
from tinytorch.layers.sequential import Sequential
from tinytorch.optimizers import SGD
from tinytorch.training import TrainingContext

# Constants
batch_size = 64
num_epochs = 5000
d1 = 10
d2 = 100
d3 = 100
d4 = 10
learning_rate = 0.01
weight_decay = 0.001
dropout_rate = 0.1

# Define model
context = TrainingContext()
block1 = Sequential(
  [
    Linear(d1, d2),
    LayerNorm(d2),  # Pre-LN: normalize BEFORE non linearity (gpt2+ good practice)
    ReLU(),
    Dropout(dropout_rate, context),  # Dropout at the end to regularize the transformed features
  ]
)
block2 = Sequential(
  [
    Linear(d2, d3),
    LayerNorm(d3),
    ReLU(),
    Dropout(dropout_rate, context),
  ]
)
model = Sequential(
  [
    block1,
    block2,
    Linear(d3, d4),
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
context.train()

for epoch in range(num_epochs):
  # Forward pass
  y_pred = model(x)
  loss = criterion(y_pred, y_true)

  # Backward pass
  grad_loss = criterion.backward()
  _ = model.backward(grad_loss)

  # Update parameters
  optimizer.step()
  model.zero_grad()

  if epoch % 100 == 0:
    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.6f}")

# Eval mode
context.eval()
y_pred_eval = model(x)
loss_eval = criterion.forward(y_pred_eval, y_true)
print(f"Eval Mode - Final Loss: {loss_eval:.6f}")
