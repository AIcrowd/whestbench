"""Parametrized correctness tests for all simulation backends."""

from __future__ import annotations

import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.generation import sample_mlp
from network_estimation.simulation import (
    output_stats as ref_output_stats,
    run_mlp as ref_run_mlp,
    run_mlp_all_layers as ref_run_mlp_all_layers,
)
from network_estimation.simulation_backends import get_available_backends, get_backend


def _make_mlp(width: int = 8, depth: int = 4, seed: int = 42) -> MLP:
    rng = np.random.default_rng(seed)
    return sample_mlp(width, depth, rng)


def _available_backend_names() -> list[str]:
    return list(get_available_backends().keys())


class TestBackendContract:
    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_name_matches_key(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        assert backend.name == backend_name

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_is_available_true(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        assert backend.__class__.is_available() is True

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_run_mlp_returns_float32(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        inputs = np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32)
        result = backend.run_mlp(mlp, inputs)
        assert result.dtype == np.float32
        assert result.shape == (16, 8)

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_run_mlp_all_layers_returns_correct_shapes(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        inputs = np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32)
        layers = backend.run_mlp_all_layers(mlp, inputs)
        assert len(layers) == 4
        for layer in layers:
            assert layer.dtype == np.float32
            assert layer.shape == (16, 8)

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_output_stats_returns_correct_shapes(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        means, final_mean, avg_var = backend.output_stats(mlp, 1000)
        assert means.dtype == np.float32
        assert means.shape == (4, 8)
        assert final_mean.dtype == np.float32
        assert final_mean.shape == (8,)
        assert isinstance(avg_var, float)


class TestExactMatch:
    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_run_mlp_matches_reference(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        inputs = np.random.default_rng(123).standard_normal((32, 8)).astype(np.float32)
        ref = ref_run_mlp(mlp, inputs)
        result = backend.run_mlp(mlp, inputs)
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_run_mlp_all_layers_matches_reference(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        inputs = np.random.default_rng(123).standard_normal((32, 8)).astype(np.float32)
        ref_layers = ref_run_mlp_all_layers(mlp, inputs)
        result_layers = backend.run_mlp_all_layers(mlp, inputs)
        assert len(result_layers) == len(ref_layers)
        for ref_layer, result_layer in zip(ref_layers, result_layers):
            np.testing.assert_allclose(result_layer, ref_layer, rtol=1e-5, atol=1e-6)


class TestStatisticalEquivalence:
    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_means_close_to_reference(self, backend_name: str) -> None:
        mlp = _make_mlp(width=64, depth=4, seed=55)
        ref_means, ref_final, ref_var = ref_output_stats(mlp, n_samples=50000)
        backend = get_backend(backend_name)
        fast_means, fast_final, fast_var = backend.output_stats(mlp, n_samples=50000)
        np.testing.assert_allclose(fast_means, ref_means, atol=0.05)
        np.testing.assert_allclose(fast_final, ref_final, atol=0.05)
        assert abs(fast_var - ref_var) < max(0.1 * abs(ref_var), 0.01)


class TestRegistry:
    def test_unknown_backend_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_numpy_always_available(self) -> None:
        backends = get_available_backends()
        assert "numpy" in backends

    def test_get_backend_default_is_numpy(self) -> None:
        import os
        old = os.environ.pop("NESTIM_BACKEND", None)
        try:
            backend = get_backend()
            assert backend.name == "numpy"
        finally:
            if old is not None:
                os.environ["NESTIM_BACKEND"] = old

    def test_get_backend_env_var(self) -> None:
        import os
        old = os.environ.get("NESTIM_BACKEND")
        os.environ["NESTIM_BACKEND"] = "numpy"
        try:
            backend = get_backend()
            assert backend.name == "numpy"
        finally:
            if old is not None:
                os.environ["NESTIM_BACKEND"] = old
            else:
                os.environ.pop("NESTIM_BACKEND", None)
