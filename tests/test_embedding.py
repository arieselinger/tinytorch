import numpy as np

from tinytorch.layers.embedding import Embedding


class TestEmbedding:
  def test_forward_shape(self) -> None:
    emb = Embedding(vocab_size=100, d_model=16)
    tokens = np.array([[1, 2, 3], [4, 5, 6]])
    out = emb.forward(tokens)
    assert out.shape == (2, 3, 16)

  def test_forward_lookup(self) -> None:
    emb = Embedding(vocab_size=10, d_model=4)
    tokens = np.array([0, 5, 0])
    out = emb.forward(tokens)
    # Same token should give same embedding
    assert np.allclose(out[0], out[2])
    assert np.allclose(out[0], emb.weights.data[0])
    assert np.allclose(out[1], emb.weights.data[5])

  def test_backward_accumulates_duplicates(self) -> None:
    """Test that duplicate tokens accumulate gradients correctly."""
    emb = Embedding(vocab_size=10, d_model=4)
    emb.weights.zero_grad()

    tokens = np.array([0, 1, 0])  # Token 0 appears twice
    grad_out = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])

    emb.forward(tokens)
    emb.backward(grad_out)

    # Token 0: grad_out[0] + grad_out[2] = [1, 0, 1, 0]
    assert np.allclose(emb.weights.grad[0], [1.0, 0, 1.0, 0])
    # Token 1: grad_out[1] = [0, 1, 0, 0]
    assert np.allclose(emb.weights.grad[1], [0, 1.0, 0, 0])
    # Other tokens: zero
    assert np.allclose(emb.weights.grad[2:], 0)
