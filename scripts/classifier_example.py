from typing import cast
import numpy as np

from tinytorch.criteria.cross_entropy import SoftmaxCrossEntropyLoss
from tinytorch.dataset import load_mnist
from tinytorch.layers.linear import Linear
from tinytorch.activations.gelu import GELU
from tinytorch.layers.layer_norm import LayerNorm
from tinytorch.layers.dropout import Dropout
from tinytorch.layers.sequence import Sequential
from tinytorch.optimizers import SGD
from tinytorch.training import TrainingContext


# Load MNIST dataset
dataset = load_mnist()
print("MNIST data loaded.")


# Prepare data
n_train = dataset.train_x.shape[0]
x_train = dataset.train_x.astype(np.float32).reshape(n_train, -1) / 255.0
y_train = dataset.train_y

n_test = dataset.test_x.shape[0]
x_test = dataset.test_x.astype(np.float32).reshape(n_test, -1) / 255.0
y_test = dataset.test_y

print("Data prepared.")

# Constants
batch_size = 64
num_epochs = 30
d_in = x_train.shape[1]  # 28 * 28 = 784
d_model = 256
d_out = 10
learning_rate = 1e-2
weight_decay = 1e-4
dropout_rate = 0.1

context = TrainingContext()
model = Sequential(
  [
    # ----- Block 1 -----
    Linear(d_in, d_model),
    LayerNorm(d_model),
    GELU(),
    Dropout(dropout_rate, context),
    # ----- Block 2 -----
    Linear(d_model, d_model),
    LayerNorm(d_model),
    GELU(),
    Dropout(dropout_rate, context),
    # ----- Logits -----
    Linear(d_model, d_out),
  ]
)
criterion = SoftmaxCrossEntropyLoss()
optimizer = SGD(model.parameters(), learning_rate, weight_decay)

print("Num parameters:", sum(p.data.size for p in model.parameters()))

# Training loop
print("Start training.")
context.train()

for epoch in range(num_epochs):
  # Shuffle training data
  indices = np.random.permutation(n_train)
  x_train_shuffled = x_train[indices]
  y_train_shuffled = y_train[indices]

  # Generate batches
  num_batches = len(x_train) // batch_size

  # Store metrics
  losses: list[float] = []
  accuracies: list[np.floating] = []

  for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size

    x = x_train_shuffled[start_idx:end_idx]
    target = y_train_shuffled[start_idx:end_idx]

    logits = model(x)
    loss = criterion(logits, target)

    # Store batch loss
    losses.append(loss.item())

    # Store batch accuracy
    preds = np.argmax(logits, axis=-1)
    acc = cast(np.floating, np.mean(preds == target))
    accuracies.append(acc)

    grad_loss = criterion.backward()
    _ = model.backward(grad_loss)

    optimizer.step()
    model.zero_grad()

  mean_loss = np.mean(losses)
  mean_acc = np.mean(accuracies)
  print(f"Epoch {epoch + 1}/{num_epochs} - mean_loss={mean_loss:.4f}, mean_acc={mean_acc:.4f}")


# Evaluation
context.eval()
print("> Start evaluation.")

logits = model.forward(x_test)
loss = criterion(logits, y_test)
preds = np.argmax(logits, axis=-1)
acc = cast(np.floating, np.mean(preds == y_test))

print(f"Evaluation - loss={loss.item():.4f}, acc={acc:.4f}")
