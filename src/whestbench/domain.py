"""Core MLP data structure and invariant checks.

This module defines the canonical in-memory representation used throughout
generation, simulation, and scoring:

- ``MLP`` stores a sequence of weight matrices plus declared width/depth metadata.

All evaluator code assumes these objects pass validation before use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import flopscope.numpy as fnp

Weights = List[fnp.ndarray]


@dataclass(frozen=True)
class MLP:
    """Validated MLP container with fixed width and layer depth.

    Attributes:
        width: Number of neurons per layer.
        depth: Number of weight matrices (layers).
        weights: Ordered list of weight matrices, each shape ``(width, width)``.
        seed: Per-MLP grader-supplied seed. Estimators using randomness should
            seed off this so their submission reproduces under regrade. Derived
            deterministically from ``ContestSpec.seed`` and the MLP index by
            the evaluator; 0 when no spec seed is provided.
        name: Human-readable per-MLP slug like ``"danielle-johnson"``. Stable
            across runs and backends at the WhestBench release's pinned
            ``faker`` version (see ``whestbench.naming``). Empty string when
            the MLP is constructed outside an evaluator bake path (e.g. in
            unit tests). Estimators may read it for log lines; the leakage
            surface is nil because it is a pure function of ``seed``.
    """

    width: int
    depth: int
    weights: Weights
    seed: int = 0
    name: str = ""

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

    @classmethod
    def from_row(
        cls,
        row: "Any",
        *,
        seed_protocol_version: str = "2.0",
    ) -> "MLP":
        """Build an MLP from a datasets.Dataset row.

        Under seed_protocol 2.0 (legacy), ``parquet["mlp_seed"]`` IS the estimator
        seed — ``mlp.seed`` returns it directly.

        Under seed_protocol 3.0 (new), ``parquet["mlp_seed"]`` is the per-MLP
        INPUT seed; ``mlp.seed`` (the estimator seed) is derived locally via
        ``int(SeedSequence(input).spawn(3)[2].generate_state(1)[0])``. This keeps
        the in-memory ``mlp.seed`` semantics identical across protocols, so
        participant estimator code is unaffected.

        Args:
            row: Dataset row dict.
            seed_protocol_version: ``"2.0"`` (legacy) or ``"3.0"`` (explicit).
                Defaults to ``"2.0"`` for callers that don't pass it (preserves
                historical behavior).

        Raises:
            ValueError: on malformed weights via MLP.validate().
        """
        weight_layers = [fnp.array(w) for w in row["weights"]]
        if not weight_layers:
            raise ValueError("MLP row has empty weights.")
        depth = len(weight_layers)
        width = weight_layers[0].shape[0] if weight_layers[0].ndim else 0

        raw_seed = int(row.get("mlp_seed", 0))
        if seed_protocol_version == "3.0":
            ss = fnp.random.SeedSequence(raw_seed).spawn(3)
            estimator_seed = int(ss[2].generate_state(1)[0])
        else:  # "2.0" or any other legacy
            estimator_seed = raw_seed

        mlp = cls(
            width=width,
            depth=depth,
            weights=weight_layers,
            seed=estimator_seed,
            name=str(row.get("mlp_name", "")),
        )
        mlp.validate()
        return mlp
