import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.simulation import (
    output_stats,
    relu,
    run_mlp,
    run_mlp_all_layers,
)


def test_relu_zeros_negatives() -> None:
    x = np.array([-1.0, 0.0, 1.0, -0.5, 2.0], dtype=np.float32)
    result = relu(x)
    np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 0.0, 2.0])


def test_run_mlp_identity_weights() -> None:
    """Identity weight matrices with ReLU should preserve positive inputs."""
    width = 4
    weights = [np.eye(width, dtype=np.float32)]
    mlp = MLP(width=width, depth=1, weights=weights)
    inputs = np.ones((2, width), dtype=np.float32)
    output = run_mlp(mlp, inputs)
    assert output.shape == (2, width)
    np.testing.assert_allclose(output, 1.0)


def test_run_mlp_all_layers_returns_per_layer() -> None:
    width = 4
    depth = 3
    weights = [np.eye(width, dtype=np.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    inputs = np.ones((5, width), dtype=np.float32)
    layers = run_mlp_all_layers(mlp, inputs)
    assert len(layers) == depth
    for layer_out in layers:
        assert layer_out.shape == (5, width)


def test_run_mlp_final_matches_all_layers_last() -> None:
    """run_mlp output should match last element of run_mlp_all_layers."""
    width = 8
    depth = 3
    rng = np.random.default_rng(42)
    weights = [(rng.standard_normal((width, width)) * 0.1).astype(np.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    inputs = rng.standard_normal((10, width)).astype(np.float32)
    final = run_mlp(mlp, inputs)
    all_layers = run_mlp_all_layers(mlp, inputs)
    np.testing.assert_array_equal(final, all_layers[-1])


def test_output_stats_returns_correct_shapes() -> None:
    width = 8
    depth = 2
    rng = np.random.default_rng(99)
    weights = [(rng.standard_normal((width, width)) * 0.1).astype(np.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    all_means, final_mean, avg_var = output_stats(mlp, n_samples=100)
    assert all_means.shape == (depth, width)
    assert final_mean.shape == (width,)
    assert isinstance(avg_var, float)
    assert avg_var >= 0.0
    np.testing.assert_allclose(final_mean, all_means[-1], atol=1e-6)
