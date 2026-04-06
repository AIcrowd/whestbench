import mechestim as me
import numpy as np

from network_estimation.estimators import (
    BaseEstimator,
    CombinedEstimator,
    CovariancePropagationEstimator,
    MeanPropagationEstimator,
)
from network_estimation.generation import sample_mlp


def _make_small_mlp():
    rng = np.random.default_rng(42)
    return sample_mlp(width=8, depth=3, rng=rng)


def test_mean_propagation_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    with me.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    result_np = np.asarray(result, dtype=np.float32)
    assert result_np.shape == (mlp.depth, mlp.width)


def test_covariance_propagation_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = CovariancePropagationEstimator()
    with me.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    result_np = np.asarray(result, dtype=np.float32)
    assert result_np.shape == (mlp.depth, mlp.width)


def test_combined_estimator_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = CombinedEstimator()
    with me.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    result_np = np.asarray(result, dtype=np.float32)
    assert result_np.shape == (mlp.depth, mlp.width)


def test_mean_propagation_all_finite() -> None:
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    with me.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    assert np.all(np.isfinite(np.asarray(result)))


def test_covariance_propagation_all_finite() -> None:
    mlp = _make_small_mlp()
    est = CovariancePropagationEstimator()
    with me.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    assert np.all(np.isfinite(np.asarray(result)))


def test_mean_propagation_nonnegative_outputs() -> None:
    """ReLU outputs are non-negative, so predicted means should be non-negative."""
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    with me.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    assert np.all(np.asarray(result) >= 0.0)


def test_combined_estimator_switches_mode() -> None:
    calls = []  # type: list[str]

    class _Mean(BaseEstimator):
        def predict(self, mlp, budget):
            calls.append("mean:{}".format(budget))
            return me.zeros((mlp.depth, mlp.width))

    class _Cov(BaseEstimator):
        def predict(self, mlp, budget):
            calls.append("cov:{}".format(budget))
            return me.ones((mlp.depth, mlp.width))

    estimator = CombinedEstimator(mean_estimator=_Mean(), covariance_estimator=_Cov())
    mlp = sample_mlp(width=4, depth=2, rng=np.random.default_rng(0))

    with me.BudgetContext(flop_budget=int(1e12)):
        low_budget = estimator.predict(mlp, budget=10)
        high_budget = estimator.predict(mlp, budget=10_000)

    np.testing.assert_allclose(np.asarray(low_budget), np.zeros((2, 4)))
    np.testing.assert_allclose(np.asarray(high_budget), np.ones((2, 4)))
    assert calls == ["mean:10", "cov:10000"]
