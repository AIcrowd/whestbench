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


@dataclass(frozen=True, slots=True)
class VectorizedCircuit:
    """Depth-major packed circuit tensors for estimator-side vectorized access.

    Attributes:
        first_idx: Parent-1 wire indices with shape ``(d, n)``.
        second_idx: Parent-2 wire indices with shape ``(d, n)``.
        coeff: Gate coefficients with shape ``(d, n, 4)`` where the last
            dimension is ``[const, first_coeff, second_coeff, product_coeff]``.
    """

    first_idx: NDArray[np.int32]
    second_idx: NDArray[np.int32]
    coeff: NDArray[np.float32]

    @property
    def const(self) -> NDArray[np.float32]:
        """Return constant terms with shape ``(d, n)``."""
        return self.coeff[:, :, 0]

    @property
    def first_coeff(self) -> NDArray[np.float32]:
        """Return first-parent linear coefficients with shape ``(d, n)``."""
        return self.coeff[:, :, 1]

    @property
    def second_coeff(self) -> NDArray[np.float32]:
        """Return second-parent linear coefficients with shape ``(d, n)``."""
        return self.coeff[:, :, 2]

    @property
    def product_coeff(self) -> NDArray[np.float32]:
        """Return bilinear coefficients with shape ``(d, n)``."""
        return self.coeff[:, :, 3]


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

    def to_vectorized(self) -> VectorizedCircuit:
        """Pack layer wiring and coefficients into depth-major tensors.

        Returns:
            A ``VectorizedCircuit`` with:
            - ``first_idx`` and ``second_idx`` of shape ``(d, n)``
            - ``coeff`` of shape ``(d, n, 4)`` where channels correspond to
              ``[const, first_coeff, second_coeff, product_coeff]``.
        """
        if self.d == 0:
            empty_idx = np.empty((0, self.n), dtype=np.int32)
            empty_coeff = np.empty((0, self.n, 4), dtype=np.float32)
            return VectorizedCircuit(first_idx=empty_idx, second_idx=empty_idx, coeff=empty_coeff)

        first_idx = np.stack([layer.first for layer in self.gates], axis=0).astype(np.int32)
        second_idx = np.stack([layer.second for layer in self.gates], axis=0).astype(np.int32)
        coeff = np.stack(
            [
                np.stack(
                    [layer.const, layer.first_coeff, layer.second_coeff, layer.product_coeff],
                    axis=-1,
                )
                for layer in self.gates
            ],
            axis=0,
        ).astype(np.float32)
        return VectorizedCircuit(first_idx=first_idx, second_idx=second_idx, coeff=coeff)

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
