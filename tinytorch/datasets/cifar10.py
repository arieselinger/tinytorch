import pickle
import urllib.request
from pathlib import Path
from typing import Callable, ClassVar, Sequence

import numpy as np

from tinytorch.datasets.base import Dataset


class CIFAR10Dataset(Dataset):
  """
  CIFAR-10 Dataset: 60k 32x32 RGB images in 10 classes

  Download from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  """

  _base_url = "https://www.cs.toronto.edu/~kriz/"
  _filename = "cifar-10-python.tar.gz"
  _class_names: ClassVar[Sequence[str]] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
  ]

  def __init__(
    self,
    root: str = ".cache",
    train: bool = True,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    transform_target: Callable[[int], int] | None = None,
  ) -> None:
    """
    CIFAR-10 Dataset

    Args:
      root: Root directory of dataset
      train: If True, get dataset from training set, otherwise from test set
      transform: Optional transform to be applied on a sample.
      transform_target: Optional transform to be applied on a target.
    """
    super().__init__(transform, transform_target)
    self.train = train

    self.cache_dir = Path(root) / "cifar10"
    self.cache_dir.mkdir(parents=True, exist_ok=True)

    self.data, self.targets = self._load_data()

    # Set mappings from class names
    self._classes = self._class_names
    self._class_to_idx = {cls_name: i for i, cls_name in enumerate(self._classes)}

  def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 data, downloading if necessary."""
    extracted_dir = self.cache_dir / "cifar-10-batches-py"

    if not extracted_dir.exists():
      self._download_and_extract()

    if self.train:
      batch_files = [extracted_dir / f"data_batch_{i}" for i in range(1, 6)]
    else:
      batch_files = [extracted_dir / "test_batch"]

    data_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for batch_file in batch_files:
      if not batch_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

      with open(batch_file, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

      # Images are stored as (10000, 3072) where 3072 = 3 * 32 * 32
      # Reshape to (10000, 3, 32, 32) for consistency
      images = batch[b"data"].astype(np.uint8).reshape(-1, 3, 32, 32)
      labels = np.array(batch[b"labels"], dtype=np.uint8)

      data_list.append(images)
      targets_list.append(labels)

    data = np.concatenate(data_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    return data, targets

  def _download_and_extract(self) -> None:
    """Download and extract CIFAR-10 dataset."""
    import tarfile

    tar_path = self.cache_dir / self._filename

    if not tar_path.exists():
      print("Downloading CIFAR-10 dataset...")
      urllib.request.urlretrieve(self._base_url + self._filename, tar_path)
      print(f"Downloaded to {tar_path}")

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
      tar.extractall(self.cache_dir)
    print("Extraction complete.")

  def __len__(self) -> int:
    return self.data.shape[0]

  def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
    sample = self.data[index]  # shape: (3, 32, 32)
    target = int(self.targets[index])

    if self.transform:
      sample = self.transform(sample)

    if self.transform_target:
      target = self.transform_target(target)

    return sample, target

  @property
  def classes(self) -> Sequence[str]:
    """Return CIFAR-10 class names."""
    return self._classes

  @property
  def class_to_idx(self) -> dict[str, int]:
    """Return mapping from class name to index."""
    return self._class_to_idx
