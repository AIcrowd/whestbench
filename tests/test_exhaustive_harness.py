import mechestim as me
import pytest

from whestbench.estimators import CovariancePropagationEstimator, MeanPropagationEstimator
from whestbench.generation import sample_mlp

pytestmark = pytest.mark.exhaustive


@pytest.mark.parametrize("seed", list(range(10)))
def test_mean_propagation_returns_finite_predictions(seed: int) -> None:
    rng = me.random.default_rng(seed)
    mlp = sample_mlp(width=4, depth=4, rng=rng)

    predicted = MeanPropagationEstimator().predict(mlp, budget=10)

    assert predicted.shape == (mlp.depth, mlp.width)
    assert me.all(me.isfinite(predicted))


@pytest.mark.parametrize("seed", list(range(10, 20)))
def test_covariance_propagation_returns_finite_predictions(seed: int) -> None:
    rng = me.random.default_rng(seed)
    mlp = sample_mlp(width=5, depth=1, rng=rng)

    predicted = CovariancePropagationEstimator().predict(mlp, budget=1000)

    assert predicted.shape == (mlp.depth, mlp.width)
    assert me.all(me.isfinite(predicted))
