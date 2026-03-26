import numpy as np
import pytest

from network_estimation.generation import sample_mlp


def test_sample_mlp_returns_valid_mlp() -> None:
    mlp = sample_mlp(width=8, depth=4)
    mlp.validate()
    assert mlp.width == 8
    assert mlp.depth == 4
    assert len(mlp.weights) == 4


def test_sample_mlp_weight_shapes() -> None:
    mlp = sample_mlp(width=16, depth=3)
    for w in mlp.weights:
        assert w.shape == (16, 16)
        assert w.dtype == np.float32


def test_sample_mlp_he_init_scale() -> None:
    """Verify weights have approximately correct He-init variance."""
    rng = np.random.default_rng(42)
    width = 256
    mlp = sample_mlp(width=width, depth=10, rng=rng)
    expected_var = 2.0 / width
    actual_var = np.var(np.concatenate([w.flatten() for w in mlp.weights]))
    assert abs(actual_var - expected_var) < 0.01 * expected_var


def test_sample_mlp_reproducible_with_rng() -> None:
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    mlp1 = sample_mlp(width=8, depth=2, rng=rng1)
    mlp2 = sample_mlp(width=8, depth=2, rng=rng2)
    for w1, w2 in zip(mlp1.weights, mlp2.weights):
        np.testing.assert_array_equal(w1, w2)


def test_sample_mlp_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="width"):
        sample_mlp(width=0, depth=1)


def test_sample_mlp_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        sample_mlp(width=4, depth=0)
