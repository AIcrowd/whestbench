import numpy as np

from network_estimation.estimators import (
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
    result = est.predict(mlp, budget=100)
    assert result.shape == (mlp.depth, mlp.width)
    assert result.dtype == np.float32


def test_covariance_propagation_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = CovariancePropagationEstimator()
    result = est.predict(mlp, budget=100)
    assert result.shape == (mlp.depth, mlp.width)
    assert result.dtype == np.float32


def test_combined_estimator_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = CombinedEstimator()
    result = est.predict(mlp, budget=100)
    assert result.shape == (mlp.depth, mlp.width)
    assert result.dtype == np.float32


def test_mean_propagation_all_finite() -> None:
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    result = est.predict(mlp, budget=100)
    assert np.all(np.isfinite(result))


def test_covariance_propagation_all_finite() -> None:
    mlp = _make_small_mlp()
    est = CovariancePropagationEstimator()
    result = est.predict(mlp, budget=100)
    assert np.all(np.isfinite(result))


def test_mean_propagation_nonnegative_outputs() -> None:
    """ReLU outputs are non-negative, so predicted means should be non-negative."""
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    result = est.predict(mlp, budget=100)
    assert np.all(result >= 0.0)
