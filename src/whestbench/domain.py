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
            seed off this so their submission reproduces under regrade. How it
            is obtained depends on the dataset's seed protocol (see
            ``whestbench.seeds``): under 2.0 the parquet ``mlp_seed`` column IS
            this value; under 3.0 it is the third ``SeedSequence`` substream of
            that column; under 4.0 it is a keyed BLAKE2b of it. On the live
            ``make_contest`` path it is derived from ``ContestSpec.seed`` and
            the MLP index. 0 when no seed is available.
        name: Human-readable per-MLP slug like ``"danielle-johnson"``. Stable
            across runs and backends at the WhestBench release's pinned
            ``faker`` version (see ``whestbench.naming``). Empty string when
            the MLP is constructed outside an evaluator bake path (e.g. in
            unit tests). Estimators may read it for log lines.

    Note:
        Under seed-protocol 4.0, ``name`` and ``seed`` are two domain-separated
        keyed derivations of the same per-MLP input seed, so neither yields the
        other and neither is computable without the dataset's key. Recognising
        an instance therefore yields identity only.

        Under 2.0 and 3.0, ``name`` is a pure function of ``seed``. That makes
        it a stable identifier for an instance across runs and submissions --
        which is what it is for, not a claim about what it protects. In
        particular, withholding ``name`` while still supplying ``seed`` is not a
        mitigation on those protocols, because anything holding ``seed``
        recomputes ``name`` via ``naming.generate_mlp_name``.
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
        seed_salt: "bytes | None" = None,
    ) -> "MLP":
        """Build an MLP from a datasets.Dataset row.

        Under seed_protocol 2.0 (legacy), ``parquet["mlp_seed"]`` IS the estimator
        seed — ``mlp.seed`` returns it directly.

        Under seed_protocol 3.0, ``parquet["mlp_seed"]`` is the per-MLP INPUT
        seed and ``mlp.seed`` is derived via
        ``int(SeedSequence(input).spawn(3)[2].generate_state(1)[0])``.

        Under seed_protocol 4.0 the column is the same INPUT seed, but
        ``mlp.seed`` is a keyed BLAKE2b of it (see ``whestbench.seeds``). That
        derivation needs the dataset's per-dataset salt, which lives in metadata
        and therefore has to be supplied by the caller — ``from_row`` never sees
        metadata. ``load_dataset`` records it, and ``iter_mlps`` / ``mlp_at`` /
        ``make_contest_from_dataset`` pass it through.

        In-memory ``mlp.seed`` semantics are identical across all three, so
        participant estimator code is unaffected.

        Args:
            row: Dataset row dict.
            seed_protocol_version: ``"2.0"`` (legacy), ``"3.0"``, or ``"4.0"``.
                Defaults to ``"2.0"`` for callers that don't pass it (preserves
                historical behavior).
            seed_salt: The dataset's protocol-4.0 salt. Required when
                ``seed_protocol_version`` is ``"4.0"``, ignored otherwise.

        Raises:
            ValueError: on malformed weights via MLP.validate(), on an
                unrecognised protocol version, or when a 4.0 row is read without
                a salt.
        """
        # float32 is the dtype contract, and it has to be asserted here rather than
        # inherited. The parquet stores float32 (Arrow `float`), but `datasets` hands
        # list columns back as nested PYTHON lists unless a format is set, and Python
        # floats are C doubles — so `fnp.array(w)` on an unformatted row silently
        # produces float64. That is a read-path artifact, not the bake: it doubled
        # every estimator's bill on anything touching the weights (flopscope prices
        # float64 at 2x float32) and doubled resident weight memory. The targets on
        # this same path were already pinned to float32 in scoring.py; the weights
        # were not. Pinning here covers every caller — iter_mlps, mlp_at,
        # make_contest_from_dataset, streaming and materialised alike — because they
        # all funnel through from_row.
        weight_layers = [fnp.array(w, dtype=fnp.float32) for w in row["weights"]]
        if not weight_layers:
            raise ValueError("MLP row has empty weights.")
        depth = len(weight_layers)
        width = weight_layers[0].shape[0] if weight_layers[0].ndim else 0

        raw_seed = int(row.get("mlp_seed", 0))
        # Validate the version here, in the one place that interprets it, so every
        # caller (make_contest_from_dataset, iter_mlps, load_mlp, direct) is guarded.
        # An unrecognised value (a typo like "3", or the whole seed_protocol dict)
        # must raise — silently treating it as legacy raw seed would reintroduce the
        # wrong-score class the explicit-protocol path exists to prevent.
        from .dataset_io import (
            SEED_PROTOCOL_VERSION,
            SEED_PROTOCOL_VERSION_V3,
            SEED_PROTOCOL_VERSION_V4,
        )

        if seed_protocol_version == SEED_PROTOCOL_VERSION_V4:
            from .seeds import derive_estimator_seed_v4

            if seed_salt is None:
                # Never fall back to an unsalted derivation: it would yield a
                # plausible-looking seed that differs from the one the dataset was
                # baked with, and scores would be silently wrong.
                raise ValueError(
                    f"seed_protocol {SEED_PROTOCOL_VERSION_V4} requires seed_salt, but none "
                    "was supplied. Load the dataset via whestbench.load_dataset so the salt "
                    "is resolved from metadata, or pass seed_salt= explicitly."
                )
            estimator_seed = derive_estimator_seed_v4(raw_seed, salt=seed_salt)
        elif seed_protocol_version == SEED_PROTOCOL_VERSION_V3:
            from .seeds import derive_estimator_seed

            estimator_seed = derive_estimator_seed(raw_seed)
        elif seed_protocol_version == SEED_PROTOCOL_VERSION:
            estimator_seed = raw_seed  # legacy: parquet mlp_seed IS the estimator seed
        else:
            raise ValueError(
                f"unsupported seed_protocol_version {seed_protocol_version!r}; expected "
                f"{SEED_PROTOCOL_VERSION!r} (legacy raw seed), "
                f"{SEED_PROTOCOL_VERSION_V3!r} (derived per-MLP seed), or "
                f"{SEED_PROTOCOL_VERSION_V4!r} (KDF per-MLP seed)"
            )

        mlp = cls(
            width=width,
            depth=depth,
            weights=weight_layers,
            seed=estimator_seed,
            name=str(row.get("mlp_name", "")),
        )
        mlp.validate()
        return mlp
