import numpy as np
import pytest

from circuit_estimation.estimators import CovariancePropagationEstimator, MeanPropagationEstimator
from circuit_estimation.generation import random_circuit
from tests.helpers import exhaustive_means

pytestmark = pytest.mark.exhaustive


@pytest.mark.parametrize("seed", list(range(10)))
def test_linearized_random_circuits_match_exhaustive_mean(seed: int) -> None:
    rng = np.random.default_rng(seed)
    circuit = random_circuit(n=4, d=4, rng=rng)

    for layer in circuit.gates:
        layer.product_coeff.fill(0.0)

    predicted = list(MeanPropagationEstimator().predict(circuit, budget=10))
    expected = exhaustive_means(circuit)

    for pred, exact in zip(predicted, expected, strict=True):
        np.testing.assert_allclose(pred, exact, atol=1e-5)


@pytest.mark.parametrize("seed", list(range(10, 20)))
def test_depth_one_covariance_matches_exhaustive_mean(seed: int) -> None:
    rng = np.random.default_rng(seed)
    circuit = random_circuit(n=5, d=1, rng=rng)

    predicted = list(CovariancePropagationEstimator().predict(circuit, budget=1000))[0]
    expected = exhaustive_means(circuit)[0]

    np.testing.assert_allclose(predicted, expected, atol=1e-5)
