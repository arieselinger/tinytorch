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


class DataLoader:
  def __init__(
    self,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
  ) -> None:
    """
    DataLoader for MNISTDataset

    Args:
      dataset: MNISTDataset instance
      batch_size: Number of samples per batch
      shuffle: Whether to shuffle the data at the start of each epoch
    """
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.indices = np.arange(len(dataset))

  def __iter__(self):
    if self.shuffle:
      np.random.shuffle(self.indices)
    self.current_idx = 0
    return self

  def __next__(self) -> tuple[np.ndarray, np.ndarray]:
    if self.current_idx >= len(self.dataset):
      raise StopIteration

    batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
    batch_samples: list[np.ndarray] = []
    batch_targets: list[int] = []

    for idx in batch_indices:
      sample, target = self.dataset[idx]
      batch_samples.append(sample)
      batch_targets.append(target)

    self.current_idx += self.batch_size

    return np.array(batch_samples), np.array(batch_targets)
