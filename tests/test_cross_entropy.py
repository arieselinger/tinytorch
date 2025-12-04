import numpy as np
import pytest

from tests.check_gradients import TOLERANCE, compare_criterion_gradients
from tinytorch.criteria.cross_entropy import SoftmaxCrossEntropyLoss


class TestSoftmaxCrossEntropyLoss:
  """Test SoftmaxCrossEntropyLoss forward and backward passes."""

  @pytest.mark.parametrize(
    "batch_size,n_classes",
    [
      (1, 3),  # Single sample, few classes
      (4, 5),  # Small batch
      (16, 10),  # Medium batch
      (32, 100),  # Large batch, many classes
    ],
  )
  def test_gradient_2d(self, batch_size: int, n_classes: int) -> None:
    """Test gradient correctness for 2D inputs (batch, classes)."""
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.random.randn(batch_size, n_classes)
    targets = np.random.randint(0, n_classes, size=(batch_size,))

    assert compare_criterion_gradients(criterion, logits, targets)

  @pytest.mark.parametrize(
    "shape,n_classes",
    [
      ((2, 3, 5), 5),  # 3D: batch x seq x classes
      ((4, 8, 10), 10),  # Larger sequence
      ((2, 4, 4, 3), 3),  # 4D: batch x height x width x classes
    ],
  )
  def test_gradient_multidim(self, shape: tuple[int, ...], n_classes: int) -> None:
    """Test gradient correctness for multi-dimensional inputs."""
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.random.randn(*shape)
    target_shape = shape[:-1]  # All dims except last
    targets = np.random.randint(0, n_classes, size=target_shape)

    assert compare_criterion_gradients(criterion, logits, targets)

  def test_loss_value(self) -> None:
    """Test that loss value is computed correctly for known example."""
    criterion = SoftmaxCrossEntropyLoss()

    # Simple case: 2 samples, 3 classes
    logits = np.array(
      [
        [2.0, 1.0, 0.1],  # Should predict class 0
        [0.5, 2.0, 0.3],  # Should predict class 1
      ]
    )
    targets = np.array([0, 1])  # Correct predictions

    # Manual calculation:
    # Sample 0: softmax([2.0, 1.0, 0.1]) ≈ [0.659, 0.242, 0.099]
    #           -log(0.659) ≈ 0.417
    # Sample 1: softmax([0.5, 2.0, 0.3]) ≈ [0.186, 0.686, 0.153]
    #           -log(0.686) ≈ 0.377
    # Mean: (0.417 + 0.377) / 2 ≈ 0.397

    loss = criterion.forward(logits, targets)

    # Check loss is positive and reasonable
    assert 0.3 < loss < 0.5, f"Loss {loss} outside expected range"

  def test_perfect_prediction(self) -> None:
    """Test loss approaches zero for perfect predictions."""
    criterion = SoftmaxCrossEntropyLoss()

    # Very confident correct predictions
    logits = np.array(
      [
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
      ]
    )
    targets = np.array([0, 1, 2])

    loss = criterion.forward(logits, targets)

    # Loss should be very small
    assert loss < 0.001, f"Loss {loss} too large for perfect predictions"

  def test_worst_prediction(self) -> None:
    """Test loss is large for worst predictions."""
    criterion = SoftmaxCrossEntropyLoss()

    # Very confident wrong predictions
    logits = np.array(
      [
        [0.0, 0.0, 10.0],  # Predicts 2, target is 0
        [10.0, 0.0, 0.0],  # Predicts 0, target is 1
      ]
    )
    targets = np.array([0, 1])

    loss = criterion.forward(logits, targets)

    # Loss should be large
    assert loss > 5.0, f"Loss {loss} too small for worst predictions"

  def test_gradient_sum_to_zero(self) -> None:
    """Test that gradients sum to zero across classes (property of softmax)."""
    criterion = SoftmaxCrossEntropyLoss()

    logits = np.random.randn(10, 5)
    targets = np.random.randint(0, 5, size=(10,))

    criterion.forward(logits, targets)
    grad = criterion.backward()

    # Sum across class dimension should be close to zero for each sample
    grad_sum = grad.sum(axis=-1)
    assert np.allclose(grad_sum, 0.0, atol=TOLERANCE), f"Gradients don't sum to zero: {grad_sum}"

  def test_numerical_stability(self) -> None:
    """Test that implementation is numerically stable with large logits."""
    criterion = SoftmaxCrossEntropyLoss()

    # Very large logits (would overflow without max subtraction)
    logits = np.array(
      [
        [1000.0, 999.0, 998.0],
        [500.0, 501.0, 499.0],
      ]
    )
    targets = np.array([0, 1])

    loss = criterion.forward(logits, targets)

    # Should not be NaN or Inf
    assert np.isfinite(loss), f"Loss is not finite: {loss}"

    # Loss should be small (correct predictions with high confidence)
    assert loss < 2.0, f"Loss {loss} unexpectedly large"
