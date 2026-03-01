import numpy as np

from circuit_estimation.generation import random_gates
from circuit_estimation.simulation import empirical_mean, run_batched
from tests.helpers import make_circuit, make_layer


def test_random_gates_shapes_and_indices() -> None:
    n = 64
    layer = random_gates(n, np.random.default_rng(123))

    assert layer.first.shape == (n,)
    assert layer.second.shape == (n,)
    assert layer.first_coeff.shape == (n,)
    assert layer.second_coeff.shape == (n,)
    assert layer.const.shape == (n,)
    assert layer.product_coeff.shape == (n,)

    assert np.issubdtype(layer.first.dtype, np.integer)
    assert np.issubdtype(layer.second.dtype, np.integer)
    assert np.all(layer.first >= 0)
    assert np.all(layer.first < n)
    assert np.all(layer.second >= 0)
    assert np.all(layer.second < n)
    assert np.all(layer.first != layer.second)


def test_random_gates_reproducible_for_same_seed() -> None:
    n = 32
    layer_a = random_gates(n, np.random.default_rng(42))
    layer_b = random_gates(n, np.random.default_rng(42))

    np.testing.assert_array_equal(layer_a.first, layer_b.first)
    np.testing.assert_array_equal(layer_a.second, layer_b.second)
    np.testing.assert_allclose(layer_a.const, layer_b.const)
    np.testing.assert_allclose(layer_a.first_coeff, layer_b.first_coeff)
    np.testing.assert_allclose(layer_a.second_coeff, layer_b.second_coeff)
    np.testing.assert_allclose(layer_a.product_coeff, layer_b.product_coeff)


def test_random_gate_polynomials_map_binary_inputs_to_binary_outputs() -> None:
    layer = random_gates(128, np.random.default_rng(7))

    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            out = (
                layer.const
                + layer.first_coeff * x
                + layer.second_coeff * y
                + layer.product_coeff * x * y
            )
            np.testing.assert_allclose(np.abs(out), 1.0, atol=1e-5)


def test_run_batched_matches_manual_layer_computation() -> None:
    layer = make_layer(
        first=[0, 1],
        second=[1, 0],
        first_coeff=[1.0, -1.0],
        second_coeff=[0.5, 0.25],
        const=[0.0, 1.0],
        product_coeff=[0.0, -0.5],
    )
    circuit = make_circuit(2, [layer])
    inputs = np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float16)

    outputs = list(run_batched(circuit, inputs))

    assert len(outputs) == 1
    expected = (
        layer.const
        + layer.first_coeff * inputs[:, layer.first]
        + layer.second_coeff * inputs[:, layer.second]
        + layer.product_coeff * inputs[:, layer.first] * inputs[:, layer.second]
    )
    np.testing.assert_allclose(outputs[0], expected)


def test_empirical_mean_exact_for_constant_circuit() -> None:
    layer = make_layer(
        first=[0, 0, 0],
        second=[1, 1, 1],
        first_coeff=[0.0, 0.0, 0.0],
        second_coeff=[0.0, 0.0, 0.0],
        const=[1.0, -1.0, 1.0],
        product_coeff=[0.0, 0.0, 0.0],
    )
    circuit = make_circuit(3, [layer])

    means = list(empirical_mean(circuit, trials=16))

    assert len(means) == 1
    np.testing.assert_allclose(means[0], np.array([1.0, -1.0, 1.0], dtype=np.float32))
