import numpy as np

from circuit_estimation.generation import random_circuit, random_gates


def test_random_gates_disallow_duplicate_inputs() -> None:
    layer = random_gates(64, np.random.default_rng(123))
    assert np.all(layer.first != layer.second)


def test_random_circuit_is_reproducible_with_seeded_rng() -> None:
    circuit_a = random_circuit(8, 3, np.random.default_rng(7))
    circuit_b = random_circuit(8, 3, np.random.default_rng(7))

    for layer_a, layer_b in zip(circuit_a.gates, circuit_b.gates):
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
