import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
  """Set random seed before each test for reproducibility."""
  np.random.seed(42)
