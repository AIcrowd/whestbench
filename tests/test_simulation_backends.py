"""Correctness tests for the mechestim simulation backend."""

from __future__ import annotations

import mechestim as me
import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.generation import sample_mlp
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
    def test_run_mlp_returns_correct_shape_and_nonnegative(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        inputs = me.array(np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32))
        result = backend.run_mlp(mlp, inputs)
        assert result.shape == (16, 8)
        assert np.all(np.asarray(result) >= 0.0), "ReLU outputs must be non-negative"

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_run_mlp_all_layers_returns_correct_shapes(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        inputs = me.array(np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32))
        layers = backend.run_mlp_all_layers(mlp, inputs)
        assert len(layers) == 4
        for layer in layers:
            assert layer.shape == (16, 8)

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_sample_layer_statistics_returns_correct_shapes(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        with me.BudgetContext(flop_budget=int(1e15)):
            means, final_mean, avg_var = backend.sample_layer_statistics(mlp, 1000)
        assert means.shape == (4, 8)
        assert final_mean.shape == (8,)
        assert isinstance(avg_var, float)
        assert avg_var >= 0.0


class TestRegistry:
    def test_unknown_backend_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_mechestim_always_available(self) -> None:
        backends = get_available_backends()
        assert "mechestim" in backends

    def test_get_backend_default_is_mechestim(self) -> None:
        import os

        old = os.environ.pop("NESTIM_BACKEND", None)
        try:
            backend = get_backend()
            assert backend.name == "mechestim"
        finally:
            if old is not None:
                os.environ["NESTIM_BACKEND"] = old

    def test_get_backend_env_var(self) -> None:
        import os

        old = os.environ.get("NESTIM_BACKEND")
        os.environ["NESTIM_BACKEND"] = "mechestim"
        try:
            backend = get_backend()
            assert backend.name == "mechestim"
        finally:
            if old is not None:
                os.environ["NESTIM_BACKEND"] = old
            else:
                os.environ.pop("NESTIM_BACKEND", None)
