import whest as we

from whestbench.estimators import (
    BaseEstimator,
    CombinedEstimator,
    CovariancePropagationEstimator,
    MeanPropagationEstimator,
)
from whestbench.generation import sample_mlp


def _make_small_mlp():
    rng = we.random.default_rng(42)
    return sample_mlp(width=8, depth=3, rng=rng)


def test_mean_propagation_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    with we.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    result_np = we.asarray(result, dtype=we.float32)
    assert result_np.shape == (mlp.depth, mlp.width)


def test_covariance_propagation_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = CovariancePropagationEstimator()
    with we.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    result_np = we.asarray(result, dtype=we.float32)
    assert result_np.shape == (mlp.depth, mlp.width)


def test_combined_estimator_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = CombinedEstimator()
    with we.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    result_np = we.asarray(result, dtype=we.float32)
    assert result_np.shape == (mlp.depth, mlp.width)


def test_mean_propagation_all_finite() -> None:
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    with we.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    assert we.all(we.isfinite(we.asarray(result)))


def test_covariance_propagation_all_finite() -> None:
    mlp = _make_small_mlp()
    est = CovariancePropagationEstimator()
    with we.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    assert we.all(we.isfinite(we.asarray(result)))


def test_mean_propagation_nonnegative_outputs() -> None:
    """ReLU outputs are non-negative, so predicted means should be non-negative."""
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    with we.BudgetContext(flop_budget=int(1e12)):
        result = est.predict(mlp, budget=100)
    assert we.all(we.asarray(result) >= 0.0)


def test_combined_estimator_switches_mode() -> None:
    calls = []  # type: list[str]

    class _Mean(BaseEstimator):
        def predict(self, mlp, budget):
            calls.append("mean:{}".format(budget))
            return we.zeros((mlp.depth, mlp.width))

    class _Cov(BaseEstimator):
        def predict(self, mlp, budget):
            calls.append("cov:{}".format(budget))
            return we.ones((mlp.depth, mlp.width))

    estimator = CombinedEstimator(mean_estimator=_Mean(), covariance_estimator=_Cov())
    mlp = sample_mlp(width=4, depth=2, rng=we.random.default_rng(0))

    with we.BudgetContext(flop_budget=int(1e12)):
        low_budget = estimator.predict(mlp, budget=10)
        high_budget = estimator.predict(mlp, budget=10_000)

    we.testing.assert_allclose(we.asarray(low_budget), we.zeros((2, 4)))
    we.testing.assert_allclose(we.asarray(high_budget), we.ones((2, 4)))
    assert calls == ["mean:10", "cov:10000"]
