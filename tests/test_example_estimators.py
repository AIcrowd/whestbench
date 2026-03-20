import numpy as np
import pytest

from network_estimation.generation import sample_mlp


@pytest.fixture
def small_mlp():
    return sample_mlp(width=8, depth=3, rng=np.random.default_rng(42))


@pytest.mark.parametrize("estimator_module", [
    "examples.estimators.random_estimator",
    "examples.estimators.mean_propagation",
    "examples.estimators.covariance_propagation",
    "examples.estimators.combined_estimator",
])
def test_example_estimator_returns_correct_shape(small_mlp, estimator_module) -> None:
    import importlib
    mod = importlib.import_module(estimator_module)
    est = mod.Estimator()
    result = est.predict(small_mlp, budget=100)
    assert result.shape == (small_mlp.depth, small_mlp.width)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
