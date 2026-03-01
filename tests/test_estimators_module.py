import numpy as np

from circuit_estimation.domain import Circuit, Layer
from circuit_estimation.estimators import (
    BaseEstimator,
    CombinedEstimator,
    CovariancePropagationEstimator,
    MeanPropagationEstimator,
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
    CovariancePropagationEstimator._clip_moments(mean, cov)
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
    predicted = np.array(
        list(MeanPropagationEstimator().predict(circuit, budget=10)),
        dtype=np.float32,
    )
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
    predicted = np.array(
        list(CovariancePropagationEstimator().predict(circuit, budget=1000)),
        dtype=np.float32,
    )
    assert predicted.shape == (1, 2)
    np.testing.assert_allclose(predicted[0], np.array([0.0, 0.0], dtype=np.float32), atol=1e-5)


def test_combined_estimator_switches_mode() -> None:
    # Budget gate must route to mean or covariance path deterministically.
    calls: list[str] = []

    class _Mean(BaseEstimator):
        def predict(self, circuit: Circuit, budget: int):
            _ = circuit
            calls.append(f"mean:{budget}")
            yield np.array([0.0], dtype=np.float32)

    class _Cov(BaseEstimator):
        def predict(self, circuit: Circuit, budget: int):
            _ = circuit
            calls.append(f"cov:{budget}")
            yield np.array([1.0], dtype=np.float32)

    estimator = CombinedEstimator(mean_estimator=_Mean(), covariance_estimator=_Cov())
    layer = Layer(
        first=np.array([0, 0], dtype=np.int32),
        second=np.array([1, 1], dtype=np.int32),
        first_coeff=np.array([0.0, 0.0], dtype=np.float32),
        second_coeff=np.array([0.0, 0.0], dtype=np.float32),
        const=np.array([1.0, 1.0], dtype=np.float32),
        product_coeff=np.array([0.0, 0.0], dtype=np.float32),
    )
    circuit = Circuit(n=2, d=1, gates=[layer])

    low_budget = np.array(list(estimator.predict(circuit, budget=10)), dtype=np.float32)
    high_budget = np.array(list(estimator.predict(circuit, budget=1000)), dtype=np.float32)

    np.testing.assert_allclose(low_budget, np.array([[0.0]], dtype=np.float32))
    np.testing.assert_allclose(high_budget, np.array([[1.0]], dtype=np.float32))
    assert calls == ["mean:10", "cov:1000"]
