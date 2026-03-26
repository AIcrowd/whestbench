import numpy as np
import pytest

from network_estimation.estimators import CovariancePropagationEstimator, MeanPropagationEstimator
from network_estimation.generation import sample_mlp
from network_estimation.simulation import sample_layer_statistics

pytestmark = pytest.mark.exhaustive


@pytest.mark.parametrize("seed", list(range(10)))
def test_mean_propagation_returns_finite_predictions(seed: int) -> None:
    rng = np.random.default_rng(seed)
    mlp = sample_mlp(width=4, depth=4, rng=rng)

    predicted = MeanPropagationEstimator().predict(mlp, budget=10)

    assert predicted.shape == (mlp.depth, mlp.width)
    assert np.all(np.isfinite(predicted))


@pytest.mark.parametrize("seed", list(range(10, 20)))
def test_covariance_propagation_returns_finite_predictions(seed: int) -> None:
    rng = np.random.default_rng(seed)
    mlp = sample_mlp(width=5, depth=1, rng=rng)

    predicted = CovariancePropagationEstimator().predict(mlp, budget=1000)

    assert predicted.shape == (mlp.depth, mlp.width)
    assert np.all(np.isfinite(predicted))
