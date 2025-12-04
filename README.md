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
- [ ] Add (residual connections)

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
- [ ] SoftmaxCrossEntropyLoss

### Optimizers

- [x] SGD
- [ ] Adam
- [ ] AdamW

## Usage Example

```python
import numpy as np

from tinytorch.layers import Linear, Sequential
from tinytorch.activations import ReLU, Softmax
from tinytorch.criteria import MSELoss
from tinytorch.optimizers import SGD

# Define model
model = Sequential([
    Linear(4, 10),
    ReLU(),
    Linear(10, 5),
    Softmax(),
])

# Setup training
criterion = MSELoss()
optimizer = SGD(model.parameters(), learning_rate=0.01, weight_decay=0.001)


# Load data
data = get_data()

# Training loop
for epoch in range(1000):
    # Load batch
    for x, y in load_batches(data):
        # Forward pass
        y_pred = model(x)
        loss = criterion(y_pred, y)

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
```
