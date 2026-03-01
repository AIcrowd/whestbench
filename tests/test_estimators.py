from __future__ import annotations

from pathlib import Path

import numpy as np

from circuit_estimation.generation import random_circuit
from circuit_estimation.loader import load_estimator_from_path
from tests.helpers import exhaustive_means, make_circuit, make_layer


def _examples_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "estimators"


def test_mean_example_matches_exhaustive_for_linear_circuit() -> None:
    estimator, _ = load_estimator_from_path(_examples_dir() / "mean_propagation.py")
    layer1 = make_layer(
        first=[0, 1, 2],
        second=[1, 2, 0],
        first_coeff=[1.0, -1.0, 0.5],
        second_coeff=[0.5, 0.25, -0.5],
        const=[0.0, 0.0, 0.0],
        product_coeff=[0.0, 0.0, 0.0],
    )
    layer2 = make_layer(
        first=[2, 0, 1],
        second=[1, 2, 0],
        first_coeff=[1.0, 0.5, -1.0],
        second_coeff=[0.0, -0.5, 0.25],
        const=[0.0, 0.0, 0.0],
        product_coeff=[0.0, 0.0, 0.0],
    )
    circuit = make_circuit(3, [layer1, layer2])

    predicted = estimator.predict(circuit, budget=10)
    expected = exhaustive_means(circuit)
    assert predicted.shape == (len(expected), circuit.n)
    for pred, exact in zip(predicted, expected, strict=True):
        np.testing.assert_allclose(pred, exact, atol=1e-6)


def test_covariance_example_depth_one_matches_exhaustive_mean() -> None:
    estimator, _ = load_estimator_from_path(_examples_dir() / "covariance_propagation.py")
    rng = np.random.default_rng(11)
    circuit = random_circuit(n=4, d=1, rng=rng)

    predicted = estimator.predict(circuit, budget=100)
    expected = exhaustive_means(circuit)

    assert predicted.shape == (1, circuit.n)
    np.testing.assert_allclose(predicted[0], expected[0], atol=1e-5)


def test_combined_example_switches_mode() -> None:
    estimator, _ = load_estimator_from_path(_examples_dir() / "combined_estimator.py")
    circuit = make_circuit(
        1,
        [
            make_layer(
                first=[0],
                second=[0],
                first_coeff=[0.0],
                second_coeff=[0.0],
                const=[0.0],
                product_coeff=[1.0],
            )
        ],
    )
    low_budget = estimator.predict(circuit, budget=10)
    high_budget = estimator.predict(circuit, budget=1000)

    np.testing.assert_allclose(low_budget, np.array([[0.0]], dtype=np.float32))
    np.testing.assert_allclose(high_budget, np.array([[1.0]], dtype=np.float32))
