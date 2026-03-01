import numpy as np
import pytest

import circuit_estimation.estimators as estimators_module
from circuit_estimation.domain import Circuit, Layer
from circuit_estimation.estimators import (
    clip,
    combined_estimator,
    covariance_propagation,
    mean_propagation,
)


def test_clip_enforces_correlation_bounds() -> None:
    # Clipping prevents impossible covariance states that break downstream math.
    mean = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    cov = np.array(
        [
            [5.0, 4.0, -4.0],
            [4.0, 5.0, 4.0],
            [-4.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    clip(mean, cov)
    assert np.all(mean <= 1.0)
    assert np.all(mean >= -1.0)
    np.testing.assert_allclose(np.diag(cov), 1.0 - mean * mean)


def test_mean_propagation_exact_for_linear_layer() -> None:
    # For linear layers, mean propagation should be exact.
    layer = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1, 0], dtype=np.int32),
        first_coeff=np.array([1.0, -1.0], dtype=np.float32),
        second_coeff=np.array([0.5, 0.25], dtype=np.float32),
        const=np.array([0.0, 0.0], dtype=np.float32),
        product_coeff=np.array([0.0, 0.0], dtype=np.float32),
    )
    circuit = Circuit(n=2, d=1, gates=[layer])
    predicted = np.array(list(mean_propagation(circuit)), dtype=np.float32)
    assert predicted.shape == (1, 2)
    np.testing.assert_allclose(predicted[0], np.array([0.0, 0.0], dtype=np.float32))


def test_covariance_propagation_depth_one_matches_linear_mean_case() -> None:
    # Covariance propagation should reduce to the same mean in linear-only circuits.
    layer = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1, 0], dtype=np.int32),
        first_coeff=np.array([1.0, -1.0], dtype=np.float32),
        second_coeff=np.array([0.5, 0.25], dtype=np.float32),
        const=np.array([0.0, 0.0], dtype=np.float32),
        product_coeff=np.array([0.0, 0.0], dtype=np.float32),
    )
    circuit = Circuit(n=2, d=1, gates=[layer])
    predicted = np.array(list(covariance_propagation(circuit)), dtype=np.float32)
    assert predicted.shape == (1, 2)
    np.testing.assert_allclose(predicted[0], np.array([0.0, 0.0], dtype=np.float32), atol=1e-5)


def test_combined_estimator_switches_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Budget gate must route to mean or covariance path deterministically.
    calls: list[str] = []

    def fake_mean(_circuit: Circuit):
        calls.append("mean")
        yield np.array([0.0], dtype=np.float32)

    def fake_cov(_circuit: Circuit):
        calls.append("cov")
        yield np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(estimators_module, "mean_propagation", fake_mean)
    monkeypatch.setattr(estimators_module, "covariance_propagation", fake_cov)

    layer = Layer(
        first=np.array([0, 0], dtype=np.int32),
        second=np.array([1, 1], dtype=np.int32),
        first_coeff=np.array([0.0, 0.0], dtype=np.float32),
        second_coeff=np.array([0.0, 0.0], dtype=np.float32),
        const=np.array([1.0, 1.0], dtype=np.float32),
        product_coeff=np.array([0.0, 0.0], dtype=np.float32),
    )
    circuit = Circuit(n=2, d=1, gates=[layer])

    low_budget = np.array(list(combined_estimator(circuit, budget=10)), dtype=np.float32)
    high_budget = np.array(list(combined_estimator(circuit, budget=1000)), dtype=np.float32)

    np.testing.assert_allclose(low_budget, np.array([[0.0]], dtype=np.float32))
    np.testing.assert_allclose(high_budget, np.array([[1.0]], dtype=np.float32))
    assert calls == ["mean", "cov"]
