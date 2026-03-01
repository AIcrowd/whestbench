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
    layer = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1, 0], dtype=np.int32),
        first_coeff=np.array([1.0, -1.0], dtype=np.float32),
        second_coeff=np.array([0.5, 0.25], dtype=np.float32),
        const=np.array([0.0, 0.0], dtype=np.float32),
        product_coeff=np.array([0.0, 0.0], dtype=np.float32),
    )
    circuit = Circuit(n=2, d=1, gates=[layer])
    predicted = list(mean_propagation(circuit))
    np.testing.assert_allclose(predicted[0], np.array([0.0, 0.0], dtype=np.float32))


def test_covariance_propagation_depth_one_matches_linear_mean_case() -> None:
    layer = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1, 0], dtype=np.int32),
        first_coeff=np.array([1.0, -1.0], dtype=np.float32),
        second_coeff=np.array([0.5, 0.25], dtype=np.float32),
        const=np.array([0.0, 0.0], dtype=np.float32),
        product_coeff=np.array([0.0, 0.0], dtype=np.float32),
    )
    circuit = Circuit(n=2, d=1, gates=[layer])
    predicted = list(covariance_propagation(circuit))
    np.testing.assert_allclose(predicted[0], np.array([0.0, 0.0], dtype=np.float32), atol=1e-5)


def test_combined_estimator_switches_mode(monkeypatch: pytest.MonkeyPatch) -> None:
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

    low_budget = list(combined_estimator(circuit, budget=10))
    high_budget = list(combined_estimator(circuit, budget=1000))

    np.testing.assert_allclose(low_budget[0], np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(high_budget[0], np.array([1.0], dtype=np.float32))
    assert calls == ["mean", "cov"]
