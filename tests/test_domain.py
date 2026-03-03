import numpy as np
import pytest

from circuit_estimation.domain import Circuit, Layer, VectorizedCircuit


def test_layer_validate_rejects_mismatched_shapes() -> None:
    # Shape alignment is a hard invariant for every layer vector.
    layer = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1], dtype=np.int32),
        first_coeff=np.array([1.0, 1.0], dtype=np.float32),
        second_coeff=np.array([1.0, 1.0], dtype=np.float32),
        const=np.array([0.0, 0.0], dtype=np.float32),
        product_coeff=np.array([0.0, 0.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="same shape"):
        layer.validate(n=2)


def test_layer_validate_rejects_out_of_bounds_indices() -> None:
    # Gate parent indices must always stay within circuit width.
    layer = Layer(
        first=np.array([0, 2], dtype=np.int32),
        second=np.array([1, 0], dtype=np.int32),
        first_coeff=np.array([1.0, 1.0], dtype=np.float32),
        second_coeff=np.array([1.0, 1.0], dtype=np.float32),
        const=np.array([0.0, 0.0], dtype=np.float32),
        product_coeff=np.array([0.0, 0.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="out of bounds"):
        layer.validate(n=2)


def test_circuit_validate_rejects_depth_mismatch() -> None:
    # Declared depth and actual layer count must agree for deterministic scoring.
    layer = Layer.identity(n=2)
    circuit = Circuit(n=2, d=2, gates=[layer])
    with pytest.raises(ValueError, match="depth"):
        circuit.validate()


def test_circuit_to_vectorized_packs_depth_major_arrays() -> None:
    layer0 = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1, 0], dtype=np.int32),
        first_coeff=np.array([1.0, 2.0], dtype=np.float32),
        second_coeff=np.array([3.0, 4.0], dtype=np.float32),
        const=np.array([5.0, 6.0], dtype=np.float32),
        product_coeff=np.array([7.0, 8.0], dtype=np.float32),
    )
    layer1 = Layer(
        first=np.array([1, 0], dtype=np.int32),
        second=np.array([0, 1], dtype=np.int32),
        first_coeff=np.array([9.0, 10.0], dtype=np.float32),
        second_coeff=np.array([11.0, 12.0], dtype=np.float32),
        const=np.array([13.0, 14.0], dtype=np.float32),
        product_coeff=np.array([15.0, 16.0], dtype=np.float32),
    )
    circuit = Circuit(n=2, d=2, gates=[layer0, layer1])

    packed = circuit.to_vectorized()

    assert isinstance(packed, VectorizedCircuit)
    assert packed.first_idx.shape == (2, 2)
    assert packed.second_idx.shape == (2, 2)
    assert packed.coeff.shape == (2, 2, 4)
    np.testing.assert_array_equal(packed.first_idx[0], layer0.first)
    np.testing.assert_array_equal(packed.second_idx[1], layer1.second)
    np.testing.assert_allclose(packed.const[0], layer0.const)
    np.testing.assert_allclose(packed.first_coeff[1], layer1.first_coeff)
    np.testing.assert_allclose(packed.second_coeff[0], layer0.second_coeff)
    np.testing.assert_allclose(packed.product_coeff[1], layer1.product_coeff)


def test_circuit_to_vectorized_handles_zero_depth() -> None:
    circuit = Circuit(n=3, d=0, gates=[])
    packed = circuit.to_vectorized()

    assert packed.first_idx.shape == (0, 3)
    assert packed.second_idx.shape == (0, 3)
    assert packed.coeff.shape == (0, 3, 4)
