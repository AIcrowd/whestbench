import numpy as np

from circuit_estimation.domain import Circuit, Layer
from circuit_estimation.simulation import empirical_mean, run_batched, run_on_random


def test_run_batched_matches_manual_layer_equation() -> None:
    # Numerical kernel should match explicit polynomial evaluation exactly.
    layer = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1, 0], dtype=np.int32),
        first_coeff=np.array([1.0, -1.0], dtype=np.float32),
        second_coeff=np.array([0.5, 0.25], dtype=np.float32),
        const=np.array([0.0, 1.0], dtype=np.float32),
        product_coeff=np.array([0.0, -0.5], dtype=np.float32),
    )
    circuit = Circuit(n=2, d=1, gates=[layer])
    inputs = np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float16)

    outputs = list(run_batched(circuit, inputs))
    expected = (
        layer.const
        + layer.first_coeff * inputs[:, layer.first]
        + layer.second_coeff * inputs[:, layer.second]
        + layer.product_coeff * inputs[:, layer.first] * inputs[:, layer.second]
    )
    np.testing.assert_allclose(outputs[0], expected)


def test_run_on_random_yields_depth_entries() -> None:
    # Runtime interface contract: one output tensor per circuit depth.
    layer = Layer.identity(n=4)
    circuit = Circuit(n=4, d=2, gates=[layer, layer])
    outputs = list(run_on_random(circuit, trials=8))
    assert len(outputs) == 2
    assert outputs[0].shape == (8, 4)


def test_empirical_mean_exact_for_constant_circuit() -> None:
    # Monte-Carlo averaging should be exact when circuit output is deterministic.
    layer = Layer(
        first=np.array([0, 0, 0], dtype=np.int32),
        second=np.array([1, 1, 1], dtype=np.int32),
        first_coeff=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        second_coeff=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        const=np.array([1.0, -1.0, 1.0], dtype=np.float32),
        product_coeff=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    circuit = Circuit(n=3, d=1, gates=[layer])
    means = list(empirical_mean(circuit, trials=16))
    np.testing.assert_allclose(means[0], np.array([1.0, -1.0, 1.0], dtype=np.float32))
