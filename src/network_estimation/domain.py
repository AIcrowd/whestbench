"""Core MLP data structure and invariant checks.

This module defines the canonical in-memory representation used throughout
generation, simulation, and scoring:

- ``MLP`` stores a sequence of weight matrices plus declared width/depth metadata.

All evaluator code assumes these objects pass validation before use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import mechestim as me

Weights = List[me.ndarray]


@dataclass(frozen=True)
class MLP:
    """Validated MLP container with fixed width and layer depth.

    Attributes:
        width: Number of neurons per layer.
        depth: Number of weight matrices (layers).
        weights: Ordered list of weight matrices, each shape ``(width, width)``.
    """

    width: int
    depth: int
    weights: Weights

    def validate(self) -> None:
        """Validate MLP metadata and weight matrix shapes.

        Raises:
            ValueError: if width/depth are invalid, if ``depth`` does not
                match ``len(weights)``, or if any weight matrix has wrong shape.
        """
        if self.width <= 0:
            raise ValueError("MLP width must be positive.")
        if self.depth <= 0:
            raise ValueError("MLP depth must be positive.")
        if len(self.weights) != self.depth:
            raise ValueError(
                f"MLP depth mismatch: declared depth={self.depth}, "
                f"got {len(self.weights)} weight matrices."
            )
        for i, w in enumerate(self.weights):
            shape = tuple(w.shape) if hasattr(w, "shape") else ()
            if shape != (self.width, self.width):
                raise ValueError(
                    f"Weight matrix {i} has shape {shape}, expected ({self.width}, {self.width})."
                )
