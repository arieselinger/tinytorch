import gzip
import urllib.request
from pathlib import Path
from typing import Callable, ClassVar, Sequence

import numpy as np

from tinytorch.datasets.base import Dataset


class MNISTDataset(Dataset):
  _base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
  _train_data = "train-images-idx3-ubyte.gz"
  _train_targets = "train-labels-idx1-ubyte.gz"
  _test_data = "t10k-images-idx3-ubyte.gz"
  _test_targets = "t10k-labels-idx1-ubyte.gz"
  _class_names: ClassVar[Sequence[str]] = [str(i) for i in range(10)]

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
    super().__init__(transform, transform_target)
    self.train = train

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

    # Set mappings from class names
    self._classes = self._class_names
    self._class_to_idx = {cls_name: i for i, cls_name in enumerate(self._classes)}

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

  @property
  def classes(self) -> Sequence[str]:
    """Return MNIST class names (digits 0-9)."""
    return self._classes

  @property
  def class_to_idx(self) -> dict[str, int]:
    """Return mapping from class name to index."""
    return self._class_to_idx

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
