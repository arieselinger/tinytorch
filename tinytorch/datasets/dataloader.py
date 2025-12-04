import numpy as np

from tinytorch.datasets.base import Dataset


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
