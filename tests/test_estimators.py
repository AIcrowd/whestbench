import pytest
import whest as we

from whestbench.estimators import (
    CombinedEstimator,
    CovariancePropagationEstimator,
    MeanPropagationEstimator,
)
from whestbench.generation import sample_mlp


@pytest.fixture
def small_mlp():
    return sample_mlp(width=4, depth=3)


def test_mean_propagation_returns_correct_shape(small_mlp) -> None:
    estimator = MeanPropagationEstimator()
    with we.BudgetContext(flop_budget=int(1e12)):
        result = estimator.predict(small_mlp, budget=1_000_000)
    result_np = we.asarray(result, dtype=we.float32)
    assert result_np.shape == (3, 4)
    assert we.all(we.isfinite(result_np))


def test_covariance_propagation_returns_correct_shape(small_mlp) -> None:
    estimator = CovariancePropagationEstimator()
    with we.BudgetContext(flop_budget=int(1e12)):
        result = estimator.predict(small_mlp, budget=1_000_000)
    result_np = we.asarray(result, dtype=we.float32)
    assert result_np.shape == (3, 4)
    assert we.all(we.isfinite(result_np))


def test_combined_estimator_routes_correctly(small_mlp) -> None:
    estimator = CombinedEstimator()
    with we.BudgetContext(flop_budget=int(1e12)):
        result = estimator.predict(small_mlp, budget=1_000_000)
    result_np = we.asarray(result, dtype=we.float32)
    assert result_np.shape == (3, 4)
