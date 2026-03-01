from pathlib import Path

import numpy as np
import pytest

from circuit_estimation.generation import random_circuit
from circuit_estimation.loader import load_estimator_from_path
from tests.helpers import exhaustive_means

pytestmark = pytest.mark.exhaustive


def _examples_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "estimators"


@pytest.mark.parametrize("seed", list(range(10)))
def test_linearized_random_circuits_match_exhaustive_mean(seed: int) -> None:
    estimator, _ = load_estimator_from_path(_examples_dir() / "mean_propagation.py")
    rng = np.random.default_rng(seed)
    circuit = random_circuit(n=4, d=4, rng=rng)

    for layer in circuit.gates:
        layer.product_coeff.fill(0.0)

    predicted = estimator.predict(circuit, budget=10)
    expected = exhaustive_means(circuit)

    for pred, exact in zip(predicted, expected):
        np.testing.assert_allclose(pred, exact, atol=1e-5)


@pytest.mark.parametrize("seed", list(range(10, 20)))
def test_depth_one_covariance_matches_exhaustive_mean(seed: int) -> None:
    estimator, _ = load_estimator_from_path(_examples_dir() / "covariance_propagation.py")
    rng = np.random.default_rng(seed)
    circuit = random_circuit(n=5, d=1, rng=rng)

    predicted = estimator.predict(circuit, budget=100)[0]
    expected = exhaustive_means(circuit)[0]

    np.testing.assert_allclose(predicted, expected, atol=1e-5)
