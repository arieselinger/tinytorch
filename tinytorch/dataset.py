from dataclasses import dataclass
import gzip
import urllib.request
from pathlib import Path
import numpy as np


@dataclass(kw_only=True, frozen=True)
class Dataset:
  train_x: np.ndarray
  train_y: np.ndarray
  test_x: np.ndarray
  test_y: np.ndarray


def load_mnist(
  cache_dir: str = ".cache/mnist",
) -> Dataset:
  """
  Download and load MNIST dataset.

  Returns:
      ((train_images, train_labels), (test_images, test_labels))
      - train_images: (60000, 28, 28) uint8 array
      - train_labels: (60000,) uint8 array
      - test_images: (10000, 28, 28) uint8 array
      - test_labels: (10000,) uint8 array
  """
  cache_path = Path(cache_dir)
  cache_path.mkdir(parents=True, exist_ok=True)

  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
  files = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
  }

  def download(filename: str) -> bytes:
    filepath = cache_path / filename
    if not filepath.exists():
      print(f"Downloading {filename}...")
      urllib.request.urlretrieve(base_url + filename, filepath)
    with gzip.open(filepath, "rb") as f:
      return f.read()

  def parse_images(data: bytes) -> np.ndarray:
    # First 16 bytes: magic number (4), num images (4), rows (4), cols (4)
    _, n, rows, cols = np.frombuffer(data[:16], dtype=">i4")
    images = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)
    return images

  def parse_labels(data: bytes) -> np.ndarray:
    # First 8 bytes: magic number (4), num labels (4)
    return np.frombuffer(data[8:], dtype=np.uint8)

  # Download and parse all files
  train_images = parse_images(download(files["train_images"]))
  train_labels = parse_labels(download(files["train_labels"]))
  test_images = parse_images(download(files["test_images"]))
  test_labels = parse_labels(download(files["test_labels"]))

  return Dataset(
    train_x=train_images,
    train_y=train_labels,
    test_x=test_images,
    test_y=test_labels,
  )
