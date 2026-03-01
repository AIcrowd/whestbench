from __future__ import annotations

from pathlib import Path

import numpy as np

from circuit_estimation.loader import load_estimator_from_path
from tests.helpers import make_circuit, make_layer


def _examples_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "estimators"


def test_mean_example_estimator_returns_depth_width_tensor() -> None:
    estimator, _ = load_estimator_from_path(_examples_dir() / "mean_propagation.py")
    circuit = make_circuit(
        3,
        [
            make_layer(
                first=[0, 1, 2],
                second=[1, 2, 0],
                first_coeff=[1.0, 0.0, 0.0],
                second_coeff=[0.0, 1.0, 0.0],
                const=[0.0, 0.0, 0.0],
                product_coeff=[0.0, 0.0, 0.0],
            ),
            make_layer(
                first=[0, 1, 2],
                second=[1, 2, 0],
                first_coeff=[0.0, 0.0, 1.0],
                second_coeff=[1.0, 0.0, 0.0],
                const=[0.0, 0.0, 0.0],
                product_coeff=[0.0, 0.0, 0.0],
            ),
        ],
    )
    predictions = estimator.predict(circuit, budget=10)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2, 3)
    np.testing.assert_allclose(predictions, np.zeros((2, 3), dtype=np.float32))


def test_covariance_example_estimator_returns_depth_width_tensor() -> None:
    estimator, _ = load_estimator_from_path(_examples_dir() / "covariance_propagation.py")
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
    predictions = estimator.predict(circuit, budget=10)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1, 1)
    np.testing.assert_allclose(predictions, np.array([[1.0]], dtype=np.float32), atol=1e-5)


def test_combined_example_switches_mode_by_budget() -> None:
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
    np.testing.assert_allclose(low_budget, np.array([[0.0]], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(high_budget, np.array([[1.0]], dtype=np.float32), atol=1e-5)
