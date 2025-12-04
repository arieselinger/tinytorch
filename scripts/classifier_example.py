from typing import cast

import numpy as np

from tinytorch.activations.gelu import GELU
from tinytorch.criteria.cross_entropy import SoftmaxCrossEntropyLoss
from tinytorch.dataset import DataLoader, MNISTDataset
from tinytorch.layers.dropout import Dropout
from tinytorch.layers.layer_norm import LayerNorm
from tinytorch.layers.linear import Linear
from tinytorch.layers.sequence import Sequential
from tinytorch.optimizers import SGD
from tinytorch.training import TrainingContext

# Load MNIST dataset
train = MNISTDataset(train=True, transform=lambda x: x.astype(np.float32).reshape(-1) / 255.0)
test = MNISTDataset(train=False, transform=lambda x: x.astype(np.float32).reshape(-1) / 255.0)
print("MNIST data loaded.")

# Constants
batch_size = 64
num_epochs = 30
d_in = train[0][0].shape[0]  # 28 * 28 = 784
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
print("Num parameters:", sum(p.data.size for p in model.parameters()))

criterion = SoftmaxCrossEntropyLoss()
optimizer = SGD(model.parameters(), learning_rate, weight_decay)

train_dataloader = DataLoader(train, batch_size, shuffle=True)
test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)

# Training loop
print("Start training.")
context.train()

for epoch in range(num_epochs):
  # Store metrics
  losses: list[float] = []
  accuracies: list[np.floating] = []

  # Iterate over batches
  for x, target in train_dataloader:
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

x_test, y_test = next(iter(test_dataloader))

logits = model.forward(x_test)
loss = criterion(logits, y_test)
preds = np.argmax(logits, axis=-1)
acc = cast(np.floating, np.mean(preds == y_test))

print(f"Evaluation - loss={loss.item():.4f}, acc={acc:.4f}")
