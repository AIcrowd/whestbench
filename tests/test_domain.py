import numpy as np
import pytest

from circuit_estimation.domain import Circuit, Layer


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
