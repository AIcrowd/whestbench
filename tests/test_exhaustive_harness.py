import numpy as np
import pytest

from circuit import random_circuit
from estimators import covariance_propagation, mean_propagation
from tests.helpers import exhaustive_means

pytestmark = pytest.mark.exhaustive


@pytest.mark.parametrize("seed", list(range(10)))
def test_linearized_random_circuits_match_exhaustive_mean(seed: int) -> None:
    rng = np.random.default_rng(seed)
    circuit = random_circuit(n=4, d=4, rng=rng)

    for layer in circuit.gates:
        layer.product_coeff.fill(0.0)

    predicted = list(mean_propagation(circuit))
    expected = exhaustive_means(circuit)

    for pred, exact in zip(predicted, expected):
        np.testing.assert_allclose(pred, exact, atol=1e-5)


@pytest.mark.parametrize("seed", list(range(10, 20)))
def test_depth_one_covariance_matches_exhaustive_mean(seed: int) -> None:
    rng = np.random.default_rng(seed)
    circuit = random_circuit(n=5, d=1, rng=rng)

    predicted = list(covariance_propagation(circuit))[0]
    expected = exhaustive_means(circuit)[0]

    np.testing.assert_allclose(predicted, expected, atol=1e-5)
