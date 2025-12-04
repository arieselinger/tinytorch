from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np


class Dataset(ABC):
  """Base class for datasets."""

  def __init__(
    self,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    transform_target: Callable[[int], int] | None = None,
  ) -> None:
    """
    Args:
      transform: Optional transform to be applied on a sample.
      transform_target: Optional transform to be applied on a target.
    """
    self.transform = transform
    self.transform_target = transform_target

  @abstractmethod
  def __len__(self) -> int:
    """Return the total number of samples."""

  @abstractmethod
  def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
    """
    Return a single sample and its target.

    Args:
      index: Index of the sample

    Returns:
      Tuple of (sample, target) where sample is np.ndarray and target is int
    """

  @property
  @abstractmethod
  def classes(self) -> Sequence[str]:
    """List of class names (index i -> class name at index i)."""

  @property
  @abstractmethod
  def class_to_idx(self) -> dict[str, int]:
    """Mapping from class name to class index."""
