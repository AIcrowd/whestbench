"""Circuit and gate sampling utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit, Layer


def random_gates(n: int, rng: np.random.Generator | None = None) -> Layer:
    """Generate one random layer of coefficients and input index pairings."""
    if n <= 1:
        raise ValueError("n must be greater than 1 to sample distinct gate inputs.")
    rng = rng or np.random.default_rng()

    is_simple = rng.choice([True, False], size=n)

    const = np.zeros(n, dtype=np.float32)
    first_coeff = np.zeros(n, dtype=np.float32)
    second_coeff = np.zeros(n, dtype=np.float32)
    product_coeff = np.zeros(n, dtype=np.float32)

    n_simple = int(np.sum(is_simple))
    if n_simple > 0:
        sign = rng.choice([-1, 1], size=n_simple).astype(np.float32)
        op_type = rng.integers(0, 4, size=n_simple)
        simple_mask = is_simple
        const[simple_mask] = (op_type == 2).astype(np.float32) * sign
        first_coeff[simple_mask] = (op_type == 0).astype(np.float32) * sign
        second_coeff[simple_mask] = (op_type == 1).astype(np.float32) * sign
        product_coeff[simple_mask] = (op_type == 3).astype(np.float32) * sign

    n_complex = n - n_simple
    if n_complex > 0:
        complex_mask = ~is_simple
        x_coeff = rng.choice([-1, 1], size=n_complex).astype(np.float32)
        y_coeff = rng.choice([-1, 1], size=n_complex).astype(np.float32)
        coeff = rng.choice([-1, 1], size=n_complex).astype(np.float32) * np.float32(0.5)
        const[complex_mask] = -coeff
        first_coeff[complex_mask] = x_coeff * coeff
        second_coeff[complex_mask] = y_coeff * coeff
        product_coeff[complex_mask] = x_coeff * y_coeff * coeff

    first: NDArray[np.int32] = rng.integers(0, n, size=n, dtype=np.int32)
    second_raw: NDArray[np.int32] = rng.integers(0, n - 1, size=n, dtype=np.int32)
    second: NDArray[np.int32] = (second_raw + (second_raw >= first).astype(np.int32)).astype(np.int32)

    layer = Layer(
        first=first,
        second=second,
        first_coeff=first_coeff,
        second_coeff=second_coeff,
        const=const,
        product_coeff=product_coeff,
    )
    layer.validate(n)
    return layer


def random_circuit(n: int, d: int, rng: np.random.Generator | None = None) -> Circuit:
    """Generate a random circuit with width ``n`` and depth ``d``."""
    if d < 0:
        raise ValueError("d must be non-negative.")
    rng = rng or np.random.default_rng()
    circuit = Circuit(n=n, d=d, gates=[random_gates(n, rng) for _ in range(d)])
    circuit.validate()
    return circuit
