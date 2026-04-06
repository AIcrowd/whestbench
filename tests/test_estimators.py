import mechestim as me
import pytest

from network_estimation.estimators import (
    CombinedEstimator,
    CovariancePropagationEstimator,
    MeanPropagationEstimator,
)
from network_estimation.generation import sample_mlp


@pytest.fixture
def small_mlp():
    return sample_mlp(width=4, depth=3)


def test_mean_propagation_returns_correct_shape(small_mlp) -> None:
    estimator = MeanPropagationEstimator()
    with me.BudgetContext(flop_budget=int(1e12)):
        result = estimator.predict(small_mlp, budget=1_000_000)
    result_np = me.asarray(result, dtype=me.float32)
    assert result_np.shape == (3, 4)
    assert me.all(me.isfinite(result_np))


def test_covariance_propagation_returns_correct_shape(small_mlp) -> None:
    estimator = CovariancePropagationEstimator()
    with me.BudgetContext(flop_budget=int(1e12)):
        result = estimator.predict(small_mlp, budget=1_000_000)
    result_np = me.asarray(result, dtype=me.float32)
    assert result_np.shape == (3, 4)
    assert me.all(me.isfinite(result_np))


def test_combined_estimator_routes_correctly(small_mlp) -> None:
    estimator = CombinedEstimator()
    with me.BudgetContext(flop_budget=int(1e12)):
        result = estimator.predict(small_mlp, budget=1_000_000)
    result_np = me.asarray(result, dtype=me.float32)
    assert result_np.shape == (3, 4)
