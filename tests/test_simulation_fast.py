"""Correctness tests for simulation_fast against the reference simulation."""

import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.simulation import (
    run_mlp as ref_run_mlp,
    run_mlp_all_layers as ref_run_mlp_all_layers,
    output_stats as ref_output_stats,
    relu as ref_relu,
)
from network_estimation.simulation_fast import (
    run_mlp as fast_run_mlp,
    run_mlp_all_layers as fast_run_mlp_all_layers,
    output_stats as fast_output_stats,
    relu as fast_relu,
)


def _make_mlp(width: int, depth: int, seed: int = 42) -> MLP:
    """Create a small deterministic MLP for testing."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2.0 / width)
    weights = [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights)


class TestReluExactMatch:
    def test_matches_reference(self) -> None:
        x = np.array([-2.0, -1.0, 0.0, 0.5, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(fast_relu(x), ref_relu(x))

    def test_all_negative(self) -> None:
        x = np.array([-5.0, -0.1, -100.0], dtype=np.float32)
        np.testing.assert_array_equal(fast_relu(x), ref_relu(x))

    def test_all_positive(self) -> None:
        x = np.array([0.1, 1.0, 99.0], dtype=np.float32)
        np.testing.assert_array_equal(fast_relu(x), ref_relu(x))


class TestRunMlpExactMatch:
    def test_small_mlp_matches_reference(self) -> None:
        mlp = _make_mlp(width=8, depth=4)
        rng = np.random.default_rng(123)
        inputs = rng.standard_normal((16, 8)).astype(np.float32)
        ref = ref_run_mlp(mlp, inputs)
        fast = fast_run_mlp(mlp, inputs)
        np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-6)

    def test_single_sample(self) -> None:
        mlp = _make_mlp(width=4, depth=2)
        inputs = np.ones((1, 4), dtype=np.float32)
        ref = ref_run_mlp(mlp, inputs)
        fast = fast_run_mlp(mlp, inputs)
        np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-6)

    def test_identity_weights(self) -> None:
        width = 4
        weights = [np.eye(width, dtype=np.float32)]
        mlp = MLP(width=width, depth=1, weights=weights)
        inputs = np.ones((2, width), dtype=np.float32)
        ref = ref_run_mlp(mlp, inputs)
        fast = fast_run_mlp(mlp, inputs)
        np.testing.assert_array_equal(fast, ref)


class TestRunMlpAllLayersExactMatch:
    def test_matches_reference(self) -> None:
        mlp = _make_mlp(width=8, depth=4)
        rng = np.random.default_rng(99)
        inputs = rng.standard_normal((16, 8)).astype(np.float32)
        ref_layers = ref_run_mlp_all_layers(mlp, inputs)
        fast_layers = fast_run_mlp_all_layers(mlp, inputs)
        assert len(fast_layers) == len(ref_layers)
        for i, (f, r) in enumerate(zip(fast_layers, ref_layers)):
            np.testing.assert_allclose(f, r, rtol=1e-5, atol=1e-6, err_msg=f"layer {i}")

    def test_last_layer_matches_run_mlp(self) -> None:
        mlp = _make_mlp(width=8, depth=3)
        rng = np.random.default_rng(7)
        inputs = rng.standard_normal((10, 8)).astype(np.float32)
        final = fast_run_mlp(mlp, inputs)
        all_layers = fast_run_mlp_all_layers(mlp, inputs)
        np.testing.assert_allclose(final, all_layers[-1], rtol=1e-5, atol=1e-6)


class TestOutputStatsStatisticalEquivalence:
    def test_means_close_to_reference(self) -> None:
        """Both paths should produce statistically similar means."""
        mlp = _make_mlp(width=8, depth=4, seed=55)
        # Different RNG streams (NumPy vs Torch) — rely on statistical convergence
        ref_means, ref_final, ref_var = ref_output_stats(mlp, n_samples=50000)
        fast_means, fast_final, fast_var = fast_output_stats(mlp, n_samples=50000)
        # Shapes must match exactly
        assert fast_means.shape == ref_means.shape
        assert fast_final.shape == ref_final.shape
        # Means should be close (different RNG streams, so use loose tolerance)
        np.testing.assert_allclose(fast_means, ref_means, atol=0.05)
        np.testing.assert_allclose(fast_final, ref_final, atol=0.05)
        # Variance should be in same ballpark
        assert abs(fast_var - ref_var) < max(0.1 * abs(ref_var), 0.01)

    def test_shapes_correct(self) -> None:
        mlp = _make_mlp(width=8, depth=2)
        means, final, var = fast_output_stats(mlp, n_samples=100)
        assert means.shape == (2, 8)
        assert final.shape == (8,)
        assert isinstance(var, float)
        assert var >= 0.0

    def test_final_mean_matches_last_layer(self) -> None:
        mlp = _make_mlp(width=8, depth=3)
        means, final, _ = fast_output_stats(mlp, n_samples=5000)
        np.testing.assert_allclose(final, means[-1], atol=1e-6)


class TestChunkBoundary:
    def test_non_divisible_n_samples(self) -> None:
        """n_samples not a multiple of chunk_size should not cause errors."""
        mlp = _make_mlp(width=8, depth=2)
        # 10007 is prime, won't divide evenly into any power-of-2 chunk
        means, final, var = fast_output_stats(mlp, n_samples=10007)
        assert means.shape == (2, 8)
        assert final.shape == (8,)
        assert isinstance(var, float)
        assert var >= 0.0

    def test_n_samples_smaller_than_chunk(self) -> None:
        """Should work even when n_samples < chunk_size."""
        mlp = _make_mlp(width=8, depth=2)
        means, final, var = fast_output_stats(mlp, n_samples=7)
        assert means.shape == (2, 8)
        assert final.shape == (8,)

    def test_n_samples_equals_one(self) -> None:
        mlp = _make_mlp(width=4, depth=2)
        means, final, var = fast_output_stats(mlp, n_samples=1)
        assert means.shape == (2, 4)
        assert final.shape == (4,)


class TestFallback:
    def test_fallback_exports_reference_functions(self) -> None:
        """When torch is unavailable, module should re-export simulation functions."""
        import importlib
        import sys
        from unittest.mock import patch

        import network_estimation.simulation as sim_mod
        import network_estimation.simulation_fast as fast_mod

        # Temporarily make torch unimportable and reload the module
        with patch.dict(sys.modules, {"torch": None}):
            importlib.reload(fast_mod)

        try:
            # After reload with torch blocked, the fallback path should have run.
            # The module's functions should be the exact same objects as simulation's.
            assert fast_mod.relu is sim_mod.relu
            assert fast_mod.run_mlp is sim_mod.run_mlp
            assert fast_mod.run_mlp_all_layers is sim_mod.run_mlp_all_layers
            assert fast_mod.output_stats is sim_mod.output_stats
            assert fast_mod._HAS_TORCH is False
        finally:
            # Restore the module to its torch-enabled state
            importlib.reload(fast_mod)
