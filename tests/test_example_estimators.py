import mechestim as me
import numpy as np
import pytest

from network_estimation.generation import sample_mlp


@pytest.fixture
def small_mlp():
    return sample_mlp(width=8, depth=3, rng=np.random.default_rng(42))


@pytest.mark.parametrize(
    "estimator_module",
    [
        "examples.estimators.random_estimator",
        "examples.estimators.mean_propagation",
        "examples.estimators.covariance_propagation",
        "examples.estimators.combined_estimator",
    ],
)
def test_example_estimator_returns_correct_shape(small_mlp, estimator_module) -> None:
    import importlib

    mod = importlib.import_module(estimator_module)
    est = mod.Estimator()
    with me.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(small_mlp, budget=100)
    result_np = np.asarray(result)
    assert result_np.shape == (small_mlp.depth, small_mlp.width)
    assert np.issubdtype(result_np.dtype, np.floating)
    assert np.all(np.isfinite(result_np))
