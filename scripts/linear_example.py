import numpy as np
from tinytorch.layers.linear import Linear

# Constants
batch_size = 12
d1 = 10
d2 = 5
d3 = 4
d4 = 17

# Define layers
l1 = Linear(d1, d2)
l2 = Linear(d2, d3)
l3 = Linear(d3, d4)

# Forward
x = np.random.randn(batch_size, d1)
y = l1(x)
z = l2(y)
w = l3(z)

# Backward
grad_w = np.random.randn(batch_size, d4)
grad_z = l3.backward(grad_w)
grad_y = l2.backward(grad_z)
grad_x = l1.backward(grad_y)

# Zero grads
l2.zero_grad()
