import numpy as np
import pytest

from tests.check_gradients import ATOL, compare_criterion_gradients
from tinytorch.criteria.cross_entropy import SoftmaxCrossEntropyLoss


class TestSoftmaxCrossEntropyLoss:
  """Test SoftmaxCrossEntropyLoss forward and backward passes."""

  @pytest.mark.parametrize(
    "shape,n_classes",
    [
      ((1, 3), 3),  # Single sample
      ((4, 5), 5),  # Small batch
      ((16, 10), 10),  # Medium batch
      ((2, 3, 5), 5),  # 3D: batch x seq x classes
      ((2, 4, 4, 3), 3),  # 4D: batch x height x width x classes
    ],
  )
  def test_gradient(self, shape: tuple[int, ...], n_classes: int) -> None:
    """Test gradient correctness for various input shapes."""
    criterion = SoftmaxCrossEntropyLoss()
    logits = np.random.randn(*shape)
    targets = np.random.randint(0, n_classes, size=shape[:-1])
    assert compare_criterion_gradients(criterion, logits, targets)

  def test_loss_value(self) -> None:
    """Test that loss value is computed correctly for known example."""
    criterion = SoftmaxCrossEntropyLoss()
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
    targets = np.array([0, 1])
    loss = criterion.forward(logits, targets)
    assert 0.3 < loss < 0.5  # Expected ~0.397

  def test_perfect_prediction(self) -> None:
    """Test loss approaches zero for perfect predictions."""
    criterion = SoftmaxCrossEntropyLoss()
    logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    targets = np.array([0, 1, 2])
    assert criterion.forward(logits, targets) < 0.001

  def test_worst_prediction(self) -> None:
    """Test loss is large for worst predictions."""
    criterion = SoftmaxCrossEntropyLoss()
    logits = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]])  # Wrong predictions
    targets = np.array([0, 1])
    assert criterion.forward(logits, targets) > 5.0

  def test_gradient_sum_to_zero(self) -> None:
    """Test that gradients sum to zero across classes (property of softmax)."""
    criterion = SoftmaxCrossEntropyLoss()
    logits = np.random.randn(10, 5)
    targets = np.random.randint(0, 5, size=(10,))
    criterion.forward(logits, targets)
    grad = criterion.backward()
    assert np.allclose(grad.sum(axis=-1), 0.0, atol=ATOL)

  def test_numerical_stability(self) -> None:
    """Test that implementation is numerically stable with large logits."""
    criterion = SoftmaxCrossEntropyLoss()
    logits = np.array([[1000.0, 999.0, 998.0], [500.0, 501.0, 499.0]])
    targets = np.array([0, 1])
    loss = criterion.forward(logits, targets)
    assert np.isfinite(loss) and loss < 2.0

  def test_ignore_index_loss(self) -> None:
    """Test that ignore_index correctly excludes positions from loss."""
    criterion = SoftmaxCrossEntropyLoss()
    logits = np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 10.0], [0.5, 2.0, 0.3]])
    targets = np.array([0, 0, 1])
    ignore_index = np.array([False, True, False])

    loss_with_ignore = criterion.forward(logits, targets, ignore_index=ignore_index)
    loss_manual = SoftmaxCrossEntropyLoss().forward(logits[[0, 2], :], targets[[0, 2]])
    assert np.isclose(loss_with_ignore, loss_manual, atol=ATOL)

  @pytest.mark.parametrize(
    "ignore_index",
    [
      np.array([False, True, False, True, False]),  # bool mask
      np.array([0, 1, 0, 1, 0]),  # int mask
    ],
  )
  def test_ignore_index_gradient(self, ignore_index: np.ndarray) -> None:
    """Test gradient correctness with ignore_index (bool and int masks)."""
    criterion = SoftmaxCrossEntropyLoss()
    logits = np.random.randn(5, 4)
    targets = np.random.randint(0, 4, size=(5,))

    assert compare_criterion_gradients(criterion, logits, targets, ignore_index=ignore_index)

    # Verify ignored positions have zero gradient
    criterion.forward(logits, targets, ignore_index=ignore_index)
    grad = criterion.backward()
    assert np.allclose(grad[1], 0.0, atol=ATOL)
    assert np.allclose(grad[3], 0.0, atol=ATOL)
