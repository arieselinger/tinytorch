import numpy as np

from tinytorch.activations.gelu import GELU
from tinytorch.criteria.cross_entropy import SoftmaxCrossEntropyLoss
from tinytorch.dataset import DataLoader
from tinytorch.datasets import normalize_and_flatten
from tinytorch.datasets.mnist import MNISTDataset
from tinytorch.layers.dropout import Dropout
from tinytorch.layers.layer_norm import LayerNorm
from tinytorch.layers.linear import Linear
from tinytorch.layers.sequence import Sequential
from tinytorch.optimizers import SGD
from tinytorch.training import TrainingContext

# Constants
batch_size = 64
num_epochs = 2
d_model = 256
learning_rate = 1e-2
weight_decay = 1e-4
dropout_rate = 0.1

# Load dataset
train = MNISTDataset(train=True, transform=normalize_and_flatten)
test = MNISTDataset(train=False, transform=normalize_and_flatten)
train_dataloader = DataLoader(train, batch_size, shuffle=True)
test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)
print("MNIST data loaded.")

d_in = train[0][0].shape[0]
d_out = len(train.classes)
print(f"Input dimension: {d_in}, Output classes: {d_out}")

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
  # Store metrics
  total_loss = 0.0
  correct = 0
  total = 0

  # Iterate over batches
  for x, target in train_dataloader:
    logits = model(x)
    loss = criterion(logits, target)

    # Accumulate loss (weighted by batch size because last batch may be smaller)
    total_loss += loss.item() * len(target)

    # Accumulate correct predictions
    preds = np.argmax(logits, axis=-1)
    correct += int((preds == target).sum())
    total += len(target)

    grad_loss = criterion.backward()
    _ = model.backward(grad_loss)

    optimizer.step()
    model.zero_grad()

  mean_loss = total_loss / total
  accuracy = correct / total
  print(f"Epoch {epoch + 1}/{num_epochs} - mean_loss={mean_loss:.4f}, acc={accuracy:.4f}")

# Evaluation
print("Start eval.")
context.eval()

x_test, y_test = next(iter(test_dataloader))

logits = model.forward(x_test)
loss = criterion(logits, y_test)
preds = np.argmax(logits, axis=-1)
accuracy = np.mean(preds == y_test, dtype=np.float32)

print(f"Evaluation - loss={loss.item():.4f}, acc={accuracy.item():.4f}")
