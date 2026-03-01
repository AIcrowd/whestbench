"""Core circuit data structures and invariant checks.

This module defines the canonical in-memory representation used throughout
generation, simulation, and scoring:

- ``Layer`` stores vectorized gate wiring and coefficients for one depth step.
- ``Circuit`` stores a sequence of layers plus declared width/depth metadata.

All evaluator code assumes these objects pass validation before use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Layer:
    """One vectorized layer of affine-bilinear gate updates.

    For each output wire ``i``, the layer applies:

    ``y[i] = const[i] + a[i] * x[first[i]] + b[i] * x[second[i]] + p[i] * x[first[i]] * x[second[i]]``

    The index arrays (``first`` and ``second``) select input wires and the
    coefficient arrays (``first_coeff``, ``second_coeff``, ``const``,
    ``product_coeff``) define the gate rule per output position.
    """

    first: NDArray[np.int32]
    second: NDArray[np.int32]
    first_coeff: NDArray[np.float32]
    second_coeff: NDArray[np.float32]
    const: NDArray[np.float32]
    product_coeff: NDArray[np.float32]

    @staticmethod
    def identity(n: int) -> "Layer":
        """Return a deterministic pass-through style layer for width ``n``.

        Each output wire copies one selected input wire via ``first`` with unit
        linear coefficient and all other coefficients set to zero. ``second``
        remains populated with valid indices so the resulting layer still
        satisfies the standard structural contract.
        """
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
        """Validate layer shape consistency and index bounds for width ``n``.

        Raises:
            ValueError: if coefficient/index vectors do not share one shape,
                or if any ``first``/``second`` index falls outside ``[0, n)``.
        """
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
    """Validated circuit container with fixed width and layer depth.

    Attributes:
        n: Wire count (width) shared by every layer.
        d: Number of transition layers in the circuit.
        gates: Ordered list of ``Layer`` instances of length ``d``.
    """

    n: int
    d: int
    gates: list[Layer]

    def validate(self) -> None:
        """Validate circuit metadata and all layer invariants.

        Raises:
            ValueError: if width/depth declarations are invalid, if ``d`` does
                not match ``len(gates)``, or if any layer violates ``Layer``
                bounds/shape requirements.
        """
        if self.n <= 0:
            raise ValueError("Circuit width n must be positive.")
        if self.d < 0:
            raise ValueError("Circuit depth d must be non-negative.")
        if len(self.gates) != self.d:
            raise ValueError("Circuit depth mismatch between d and number of gates.")

        for layer in self.gates:
            layer.validate(self.n)
