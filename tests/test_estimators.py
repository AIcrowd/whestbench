import numpy as np
import pytest

import circuit_estimation.estimators as estimators
from circuit_estimation.estimators import (
    clip,
    combined_estimator,
    covariance_propagation,
    mean_propagation,
    one_v_two_covariance,
    two_v_two_covariance,
)
from circuit_estimation.generation import random_circuit
from tests.helpers import exhaustive_means, make_circuit, make_layer


def test_mean_propagation_exact_for_linear_circuit() -> None:
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

    predicted = np.array(list(mean_propagation(circuit)), dtype=np.float32)
    expected = exhaustive_means(circuit)

    assert predicted.shape == (len(expected), circuit.n)
    for pred, exact in zip(predicted, expected, strict=True):
        np.testing.assert_allclose(pred, exact, atol=1e-6)


def test_one_v_two_covariance_matches_manual_formula() -> None:
    a = np.array([0, 2], dtype=np.int32)
    b = np.array([1, 0], dtype=np.int32)
    c = np.array([2, 1], dtype=np.int32)
    x_mean = np.array([0.1, -0.3, 0.4], dtype=np.float32)
    x_cov = np.array(
        [
            [0.2, -0.1, 0.05],
            [-0.1, 0.5, 0.2],
            [0.05, 0.2, 0.8],
        ],
        dtype=np.float32,
    )

    observed = one_v_two_covariance(a, b, c, x_cov, x_mean)

    expected = np.zeros((len(a), len(b)), dtype=np.float32)
    for i in range(len(a)):
        for j in range(len(b)):
            expected[i, j] = x_mean[b[j]] * x_cov[a[i], c[j]] + x_mean[c[j]] * x_cov[a[i], b[j]]
    np.testing.assert_allclose(observed, expected)


def test_two_v_two_covariance_matches_manual_formula() -> None:
    a = np.array([0, 1], dtype=np.int32)
    b = np.array([2, 0], dtype=np.int32)
    c = np.array([1, 2], dtype=np.int32)
    d = np.array([0, 1], dtype=np.int32)
    mean = np.array([0.2, -0.1, 0.3], dtype=np.float32)
    cov = np.array(
        [
            [0.5, 0.1, -0.2],
            [0.1, 0.4, 0.05],
            [-0.2, 0.05, 0.7],
        ],
        dtype=np.float32,
    )

    observed = two_v_two_covariance(a, b, c, d, cov, mean)

    expected = np.zeros((len(a), len(c)), dtype=np.float32)
    for i in range(len(a)):
        for j in range(len(c)):
            mu_a = mean[a[i]]
            mu_b = mean[b[i]]
            mu_c = mean[c[j]]
            mu_d = mean[d[j]]
            expected[i, j] = (
                (mu_a * mu_c) * cov[b[i], d[j]]
                + (mu_a * mu_d) * cov[b[i], c[j]]
                + (mu_b * mu_c) * cov[a[i], d[j]]
                + (mu_b * mu_d) * cov[a[i], c[j]]
            )
    np.testing.assert_allclose(observed, expected)


def test_clip_limits_mean_and_covariance() -> None:
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

    std = np.sqrt(np.clip(1.0 - mean * mean, 0.0, None))
    limit = np.outer(std, std)
    assert np.all(cov <= limit + 1e-6)
    assert np.all(cov >= -limit - 1e-6)


def test_covariance_propagation_depth_one_matches_exhaustive_mean() -> None:
    rng = np.random.default_rng(11)
    circuit = random_circuit(n=4, d=1, rng=rng)

    predicted = np.array(list(covariance_propagation(circuit)), dtype=np.float32)
    expected = exhaustive_means(circuit)

    assert predicted.shape == (1, circuit.n)
    np.testing.assert_allclose(predicted[0], expected[0], atol=1e-5)


def test_combined_estimator_switches_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_mean(_circuit):
        calls.append("mean")
        yield np.array([0.0], dtype=np.float32)

    def fake_cov(_circuit):
        calls.append("cov")
        yield np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(estimators, "mean_propagation", fake_mean)
    monkeypatch.setattr(estimators, "covariance_propagation", fake_cov)

    circuit = make_circuit(
        2,
        [
            make_layer(
                first=[0, 0],
                second=[1, 1],
                first_coeff=[0.0, 0.0],
                second_coeff=[0.0, 0.0],
                const=[1.0, 1.0],
                product_coeff=[0.0, 0.0],
            )
        ],
    )

    low_budget = np.array(list(combined_estimator(circuit, budget=10)), dtype=np.float32)
    high_budget = np.array(list(combined_estimator(circuit, budget=1000)), dtype=np.float32)

    np.testing.assert_allclose(low_budget, np.array([[0.0]], dtype=np.float32))
    np.testing.assert_allclose(high_budget, np.array([[1.0]], dtype=np.float32))
    assert calls == ["mean", "cov"]
