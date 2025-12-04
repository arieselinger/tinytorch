# TinyTorch 🦕

_A tiny deep learning framework built from scratch using only NumPy._

Every layer, gradient, and transformation is written manually: no autograd, no hidden magic! Built for learning and interview prep.

- [Roadmap](#roadmap)
- [Usage Example](#usage-example)
- [Quick Start](#quick-start)

## Roadmap

### Base Components

- [x] `Module` base class with forward/backward/parameters interface
- [x] `Parameter` wrapper for gradient tracking
- [x] `Sequential` container for chaining single input layers

### Layers

- [x] `Linear`
- [x] `Dropout`
- [x] `LayerNorm`
- [x] `Add` (residual connections)

### Attention / Transformers

- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Attention
- [ ] Self-Attention Block (pre-norm)
- [ ] Feed-Forward Block (pre-norm)
- [ ] Transformer Encoder Block

### Activations

- [x] ReLU
- [x] Sigmoid
- [x] Softmax
- [x] GELU

### Loss Functions

- [x] MSELoss
- [x] SoftmaxCrossEntropyLoss

### Optimizers

- [x] SGD
- [ ] Adam
- [ ] AdamW

### Datasets & DataLoaders

- [x] Dataset (base class)
- [x] DataLoader (batching, shuffling)
- [x] MNISTDataset
- [x] CIFAR10Dataset

### Scripts / Examples

- [x] Linear regression example
- [x] Sequential model example
- [x] SGD optimizer example
- [x] Classifier example on MNIST dataset (98%+ accuracy)

## Usage Example

```python
from tinytorch.activations.relu import ReLU
from tinytorch.criteria.cross_entropy import SoftmaxCrossEntropyLoss
from tinytorch.datasets import normalize_and_flatten
from tinytorch.datasets.dataloader import DataLoader
from tinytorch.datasets.mnist import MNISTDataset
from tinytorch.layers.layer_norm import LayerNorm
from tinytorch.layers.linear import Linear
from tinytorch.layers.sequential import Sequential
from tinytorch.optimizers import SGD

# Define model
model = Sequential(
  [
    Linear(784, 256),
    LayerNorm(256),
    ReLU(),
    Linear(256, 10),
  ]
)

# Load data
dataset = MNISTDataset(train=True, transform=normalize_and_flatten)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Setup training
criterion = SoftmaxCrossEntropyLoss()
optimizer = SGD(model.parameters(), learning_rate=0.01, weight_decay=0.001)

# Training loop
for epoch in range(10):
  # Load batch
  for x, targets in dataloader:
    # Forward pass
    logits = model(x)
    loss = criterion(logits, targets)

    # Backward pass
    grad_loss = criterion.backward()
    model.backward(grad_loss)

    # Update weights
    optimizer.step()
    model.zero_grad()
```

## Quick Start

Source the setup script to install `poetry`, set up the virtual env, and install dependencies:

```bash
. setup
```

Run examples:

```bash
python scripts/linear_example.py
python scripts/sequential_example.py
python scripts/sgd_example.py
python scripts/classifier_example.py     # MNIST: 98%+ accuracy
```
