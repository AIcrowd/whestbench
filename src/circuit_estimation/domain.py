"""Domain entities and invariants for random Boolean-like circuit layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Layer:
    """A single layer of affine-bilinear gate coefficients over indexed inputs."""

    first: NDArray[np.int32]
    second: NDArray[np.int32]
    first_coeff: NDArray[np.float32]
    second_coeff: NDArray[np.float32]
    const: NDArray[np.float32]
    product_coeff: NDArray[np.float32]

    @staticmethod
    def identity(n: int) -> "Layer":
        """Return an identity-like layer where each output copies its indexed first input."""
        first = np.arange(n, dtype=np.int32)
        second = np.roll(first, -1).astype(np.int32)
        return Layer(
            first=first,
            second=second,
            first_coeff=np.ones(n, dtype=np.float32),
            second_coeff=np.zeros(n, dtype=np.float32),
            const=np.zeros(n, dtype=np.float32),
            product_coeff=np.zeros(n, dtype=np.float32),
        )

    def validate(self, n: int) -> None:
        """Validate shape and index bounds relative to a circuit width ``n``."""
        shapes = (
            self.first.shape,
            self.second.shape,
            self.first_coeff.shape,
            self.second_coeff.shape,
            self.const.shape,
            self.product_coeff.shape,
        )
        if len(set(shapes)) != 1:
            raise ValueError("All layer vectors must have the same shape.")

        if np.any(self.first < 0) or np.any(self.first >= n):
            raise ValueError("first indices are out of bounds.")
        if np.any(self.second < 0) or np.any(self.second >= n):
            raise ValueError("second indices are out of bounds.")


@dataclass(slots=True)
class Circuit:
    """Circuit-level container for layer sequence and declared dimensions."""

    n: int
    d: int
    gates: list[Layer]

    def validate(self) -> None:
        """Validate depth and delegate per-layer validation."""
        if self.n <= 0:
            raise ValueError("Circuit width n must be positive.")
        if self.d < 0:
            raise ValueError("Circuit depth d must be non-negative.")
        if len(self.gates) != self.d:
            raise ValueError("Circuit depth mismatch between d and number of gates.")

        for layer in self.gates:
            layer.validate(self.n)
