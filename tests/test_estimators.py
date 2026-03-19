import numpy as np

from network_estimation.estimators import (
    BaseEstimator,
    CombinedEstimator,
    CovariancePropagationEstimator,
    MeanPropagationEstimator,
)
from network_estimation.generation import sample_mlp


def test_mean_propagation_returns_correct_shape() -> None:
    mlp = sample_mlp(width=4, depth=3, rng=np.random.default_rng(42))
    estimator = MeanPropagationEstimator()
    predicted = estimator.predict(mlp, budget=10)

    assert predicted.shape == (mlp.depth, mlp.width)
    assert predicted.dtype == np.float32


def test_covariance_propagation_returns_correct_shape() -> None:
    mlp = sample_mlp(width=4, depth=3, rng=np.random.default_rng(42))
    estimator = CovariancePropagationEstimator()
    predicted = estimator.predict(mlp, budget=1000)

    assert predicted.shape == (mlp.depth, mlp.width)
    assert predicted.dtype == np.float32


def test_mean_propagation_produces_finite_values() -> None:
    mlp = sample_mlp(width=8, depth=4, rng=np.random.default_rng(7))
    estimator = MeanPropagationEstimator()
    predicted = estimator.predict(mlp, budget=10)

    assert np.all(np.isfinite(predicted))


def test_covariance_propagation_produces_finite_values() -> None:
    mlp = sample_mlp(width=8, depth=4, rng=np.random.default_rng(7))
    estimator = CovariancePropagationEstimator()
    predicted = estimator.predict(mlp, budget=1000)

    assert np.all(np.isfinite(predicted))


def test_combined_estimator_switches_mode() -> None:
    calls = []  # type: list[str]

    class _Mean(BaseEstimator):
        def predict(self, mlp, budget):
            calls.append("mean:{}".format(budget))
            return np.zeros((mlp.depth, mlp.width), dtype=np.float32)

    class _Cov(BaseEstimator):
        def predict(self, mlp, budget):
            calls.append("cov:{}".format(budget))
            return np.ones((mlp.depth, mlp.width), dtype=np.float32)

    estimator = CombinedEstimator(mean_estimator=_Mean(), covariance_estimator=_Cov())
    mlp = sample_mlp(width=4, depth=2, rng=np.random.default_rng(0))

    low_budget = estimator.predict(mlp, budget=10)
    high_budget = estimator.predict(mlp, budget=1000)

    np.testing.assert_allclose(low_budget, np.zeros((2, 4), dtype=np.float32))
    np.testing.assert_allclose(high_budget, np.ones((2, 4), dtype=np.float32))
    assert calls == ["mean:10", "cov:1000"]
