import whest as we

from whestbench.domain import MLP
from whestbench.generation import sample_mlp
from whestbench.simulation import (
    relu,
    run_mlp,
    run_mlp_all_layers,
    sample_layer_statistics,
)


def _make_mlp(width: int = 8, depth: int = 2, seed: int = 42) -> MLP:
    rng = we.random.default_rng(seed)
    return sample_mlp(width, depth, rng)


def test_relu_zeros_negatives() -> None:
    x = we.array([-1.0, 0.0, 1.0, -0.5, 2.0], dtype=we.float32)
    result = relu(x)
    we.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 0.0, 2.0])


def test_run_mlp_identity_weights() -> None:
    """Identity weight matrices with ReLU should preserve positive inputs."""
    width = 4
    weights = [we.eye(width, dtype=we.float32)]
    mlp = MLP(width=width, depth=1, weights=weights)
    inputs = we.ones((2, width), dtype=we.float32)
    output = run_mlp(mlp, inputs)
    assert output.shape == (2, width)
    we.testing.assert_allclose(output, 1.0)


def test_run_mlp_all_layers_returns_per_layer() -> None:
    width = 4
    depth = 3
    weights = [we.eye(width, dtype=we.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    inputs = we.ones((5, width), dtype=we.float32)
    layers = run_mlp_all_layers(mlp, inputs)
    assert len(layers) == depth
    for layer_out in layers:
        assert layer_out.shape == (5, width)


def test_run_mlp_final_matches_all_layers_last() -> None:
    """run_mlp output should match last element of run_mlp_all_layers."""
    width = 8
    depth = 3
    rng = we.random.default_rng(42)
    weights = [(rng.standard_normal((width, width)) * 0.1).astype(we.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    inputs = rng.standard_normal((10, width)).astype(we.float32)
    final = run_mlp(mlp, inputs)
    all_layers = run_mlp_all_layers(mlp, inputs)
    we.testing.assert_array_equal(final, all_layers[-1])


def test_sample_layer_statistics_returns_correct_shapes() -> None:
    width = 8
    depth = 2
    rng = we.random.default_rng(99)
    weights = [(rng.standard_normal((width, width)) * 0.1).astype(we.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    all_means, final_mean, avg_var = sample_layer_statistics(mlp, n_samples=100)
    assert all_means.shape == (depth, width)
    assert final_mean.shape == (width,)
    assert isinstance(avg_var, float)
    assert avg_var >= 0.0
    we.testing.assert_allclose(final_mean, all_means[-1], atol=1e-6)


def test_sample_layer_statistics_handles_large_sample_count() -> None:
    """Verify chunked sampling produces correct shapes with n_samples > chunk_size."""
    mlp = _make_mlp(width=8, depth=2)
    with we.BudgetContext(flop_budget=int(1e15)):
        all_means, final_mean, avg_var = sample_layer_statistics(mlp, n_samples=3000)
    assert all_means.shape == (2, 8)
    assert final_mean.shape == (8,)
    assert isinstance(avg_var, float)
    assert avg_var >= 0.0
