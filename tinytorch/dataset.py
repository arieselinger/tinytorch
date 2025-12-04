import gzip
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np


class MNISTDataset:
  _base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
  _train_data = "train-images-idx3-ubyte.gz"
  _train_targets = "train-labels-idx1-ubyte.gz"
  _test_data = "t10k-images-idx3-ubyte.gz"
  _test_targets = "t10k-labels-idx1-ubyte.gz"

  def __init__(
    self,
    root: str = ".cache",
    train: bool = True,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    transform_target: Callable[[int], int] | None = None,
  ) -> None:
    """
    MNIST Dataset

    Args:
      root: Root directory of dataset
      train: If True, get dataset from training set, otherwise from test set
      transform: Optional transform to be applied on a sample.
      transform_target: Optional transform to be applied on a target.
    """
    self.train = train
    self.transform = transform
    self.transform_target = transform_target

    self.cache_dir = Path(root) / "mnist"
    self.cache_dir.mkdir(parents=True, exist_ok=True)

    if train:
      data_file = self._train_data
      targets_file = self._train_targets
    else:
      data_file = self._test_data
      targets_file = self._test_targets

    self.data = self.parse_images(self.download_or_get(data_file))
    self.targets = self.parse_labels(self.download_or_get(targets_file))

  def download_or_get(self, filename: str) -> bytes:
    filepath = self.cache_dir / filename
    if not filepath.exists():
      print(f"Downloading {filename}.")
      urllib.request.urlretrieve(self._base_url + filename, filepath)
    with gzip.open(filepath, "rb") as f:
      return f.read()

  def parse_images(self, data: bytes) -> np.ndarray:
    # First 16 bytes: magic number (4), num images (4), rows (4), cols (4)
    _, n, rows, cols = np.frombuffer(data[:16], dtype=">i4")
    images = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)
    return images

  def parse_labels(self, data: bytes) -> np.ndarray:
    # First 8 bytes: magic number (4), num labels (4)
    return np.frombuffer(data[8:], dtype=np.uint8)

  def __len__(self) -> int:
    return self.data.shape[0]

  def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
    sample = self.data[index]
    target = int(self.targets[index])

    if self.transform:
      sample = self.transform(sample)

    if self.transform_target:
      target = self.transform_target(target)

    return sample, target


class DataLoader:
  def __init__(
    self,
    dataset: MNISTDataset,
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
