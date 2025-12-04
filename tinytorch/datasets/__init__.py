import numpy as np


def normalize_and_flatten(x: np.ndarray) -> np.ndarray:
  """Normalize image to [0, 1] and flatten to 1D."""
  return x.astype(np.float32).reshape(-1) / 255.0
