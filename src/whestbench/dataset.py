"""Create, save, and load pre-computed evaluation datasets."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np  # needed for np.savez, np.load (file I/O)

from .domain import MLP
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint
from .naming import assign_unique_names
from .scoring import _normalize_sampling_budget_breakdown
from .simulation import sample_layer_statistics, sample_layer_statistics_chunk_count

# 2.4 added the `mlp_names` array (per-MLP human-readable slug, see
# `whestbench.naming`). Names are a pure function of `mlp_seeds`, so 2.3
# files load cleanly by synthesizing names on the fly.
SCHEMA_VERSION = "2.4"
SEED_PROTOCOL_NAME = "whestbench_seedsequence_hierarchy"
SEED_PROTOCOL_VERSION = "2.0"


def dataset_file_hash(path: "Path | str") -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DatasetBundle:
    metadata: Dict[str, Any]
    mlps: List[MLP]
    all_layer_means: fnp.ndarray
    final_means: fnp.ndarray
    avg_variances: List[float]
    sampling_budget_breakdowns: List[Dict[str, Any]] | None = None

    @property
    def n_mlps(self) -> int:
        return len(self.mlps)


def create_dataset(
    *,
    n_mlps: int,
    n_samples: int,
    width: int,
    depth: int,
    seed: Optional[int] = None,
    output_path: "Path | str",
    progress: Optional[Any] = None,
) -> Path:
    """Generate MLPs, compute ground truth, and save to .npz."""
    output_path = Path(output_path)
    seed_sequence = (
        fnp.random.SeedSequence() if seed is None else fnp.random.SeedSequence(int(seed))
    )
    stream_seed = seed_sequence.spawn(3 * n_mlps)

    mlps: List[MLP] = []
    for i in range(n_mlps):
        weight_stream = fnp.random.default_rng(stream_seed[3 * i])
        estimator_seed_i = int(stream_seed[3 * i + 2].generate_state(1)[0])
        mlps.append(sample_mlp(width, depth, weight_stream, seed=estimator_seed_i))
        if progress is not None:
            progress({"phase": "generating", "completed": i + 1, "total": n_mlps})

    # Attach deterministic human-readable names derived from each MLP's seed.
    # Names are a pure function of `mlp.seed`, so they reproduce across runs
    # and across CPU/GPU backends at the pinned faker version.
    mlp_names_list = assign_unique_names([m.seed for m in mlps])
    mlps = [dataclasses.replace(m, name=n) for m, n in zip(mlps, mlp_names_list)]

    # Pack weight matrices: shape (n_mlps, depth, width, width)
    weights_array = np.stack([np.stack(mlp.weights) for mlp in mlps]).astype(np.float32)

    # Compute ground truth
    all_means_list: List[fnp.ndarray] = []
    final_means_list: List[fnp.ndarray] = []
    avg_variances: List[float] = []
    sampling_budget_breakdowns: List[Dict[str, Any]] = []
    chunks_per_mlp = sample_layer_statistics_chunk_count(width, n_samples)
    total_sampling_chunks = n_mlps * chunks_per_mlp
    for i, mlp in enumerate(mlps):
        sample_stream = fnp.random.default_rng(stream_seed[3 * i + 1])

        def _on_sampling_chunk(
            event: Dict[str, Any],
            *,
            mlp_index: int = i + 1,
            mlp_name: str = mlps[i].name,
            chunk_offset: int = i * chunks_per_mlp,
        ) -> None:
            if progress is None:
                return
            local_completed = int(event.get("completed", 0))
            local_total = int(event.get("total", chunks_per_mlp))
            progress(
                {
                    "phase": "sampling",
                    "completed": chunk_offset + local_completed,
                    "total": total_sampling_chunks,
                    "mlp_index": mlp_index,
                    "mlp_name": mlp_name,
                    "n_mlps": n_mlps,
                    "mlp_completed": local_completed,
                    "mlp_total": local_total,
                    "unit": "chunks",
                }
            )

        with flops.BudgetContext(flop_budget=int(1e15), quiet=True) as sampling_budget:
            with flops.namespace("sampling"):
                with flops.namespace("sample_layer_statistics"):
                    all_means, final_mean, avg_var = sample_layer_statistics(
                        mlp,
                        n_samples,
                        rng=sample_stream,
                        progress=_on_sampling_chunk if progress is not None else None,
                    )
        normalized_sampling = _normalize_sampling_budget_breakdown(
            sampling_budget.summary_dict(by_namespace=True)
        )
        if normalized_sampling is not None:
            sampling_budget_breakdowns.append(normalized_sampling)
        # The triple is always bound: flopscope's namespace `__exit__` returns
        # False at runtime (never suppresses), but pyright reads the `-> bool`
        # annotation conservatively.
        all_means_list.append(fnp.asarray(all_means, dtype=fnp.float32))  # pyright: ignore[reportPossiblyUnboundVariable]
        final_means_list.append(fnp.asarray(final_mean, dtype=fnp.float32))  # pyright: ignore[reportPossiblyUnboundVariable]
        avg_variances.append(avg_var)  # pyright: ignore[reportPossiblyUnboundVariable]

    all_layer_means = np.stack(all_means_list).astype(np.float32)
    final_means = np.stack(final_means_list).astype(np.float32)

    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend": "flopscope",
        "seed_protocol": {
            "name": SEED_PROTOCOL_NAME,
            "version": SEED_PROTOCOL_VERSION,
            "seeded": seed is not None,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed_sequence.entropy),
        "n_mlps": n_mlps,
        "n_samples": n_samples,
        "width": width,
        "depth": depth,
        "hardware": collect_hardware_fingerprint(),
    }

    mlp_seeds = np.array([m.seed for m in mlps], dtype=np.int64)
    # U64 is comfortably above the longest realistic slug (faker names are
    # generally < 20 chars; collision suffixes add a few more).
    mlp_names_array = np.array([m.name for m in mlps], dtype="U64")

    np.savez(
        output_path,
        metadata=np.array(json.dumps(metadata)),
        weights=weights_array,
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=np.array(avg_variances, dtype=np.float64),
        sampling_budget_breakdowns=np.array(json.dumps(sampling_budget_breakdowns)),
        mlp_seeds=mlp_seeds,
        mlp_names=mlp_names_array,
    )
    return output_path


def load_dataset(path: "Path | str") -> DatasetBundle:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))

    if "schema_version" not in metadata:
        raise ValueError("Invalid dataset: missing schema_version.")

    seed_proto = metadata.get("seed_protocol") or {}
    protocol_version = str(seed_proto.get("version", "")) if isinstance(seed_proto, dict) else ""
    if protocol_version and protocol_version != SEED_PROTOCOL_VERSION:
        raise ValueError(
            f"Incompatible dataset seed_protocol version: file has {protocol_version!r}, "
            f"this whestbench requires {SEED_PROTOCOL_VERSION!r}. "
            f"Re-bake the dataset with `whest create-dataset`."
        )

    weights_array = data["weights"].astype(np.float32)
    all_layer_means = data["all_layer_means"].astype(np.float32)
    final_means = data["final_means"].astype(np.float32)
    avg_variances = data["avg_variances"].astype(np.float64).tolist()
    sampling_budget_breakdowns: List[Dict[str, Any]] | None = None
    if "sampling_budget_breakdowns" in data.files:
        raw_sampling_breakdowns = json.loads(str(data["sampling_budget_breakdowns"]))
        if isinstance(raw_sampling_breakdowns, list):
            sampling_budget_breakdowns = [
                item for item in raw_sampling_breakdowns if isinstance(item, dict)
            ]

    n_mlps = int(weights_array.shape[0])
    depth = int(weights_array.shape[1])
    width = int(weights_array.shape[2])

    if "mlp_seeds" in data.files:
        mlp_seeds = data["mlp_seeds"].astype(np.int64).tolist()
    elif protocol_version == SEED_PROTOCOL_VERSION:
        raise ValueError(
            "Dataset is at SEED_PROTOCOL_VERSION='2.0' but missing 'mlp_seeds' array; "
            "the file appears corrupted. Re-bake with `whest create-dataset`."
        )
    else:
        # Legacy datasets pre-dating seed_protocol carry no per-MLP seeds; default to 0.
        mlp_seeds = [0] * n_mlps

    # Schema 2.4 introduced per-MLP names. For older files (no `mlp_names`),
    # synthesize from `mlp_seeds` — names are a pure function of seed, so the
    # synthesized values match what a fresh 2.4 bake of the same seeds would
    # produce. No separate "placeholder" fallback.
    if "mlp_names" in data.files:
        mlp_names = [str(s) for s in data["mlp_names"].tolist()]
    else:
        mlp_names = assign_unique_names(list(mlp_seeds))

    mlps: List[MLP] = []
    for i in range(n_mlps):
        layer_weights = [fnp.array(weights_array[i, j]) for j in range(depth)]
        mlp = MLP(
            width=width,
            depth=depth,
            weights=layer_weights,
            seed=int(mlp_seeds[i]),
            name=mlp_names[i],
        )
        mlp.validate()
        mlps.append(mlp)

    return DatasetBundle(
        metadata=metadata,
        mlps=mlps,
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=avg_variances,
        sampling_budget_breakdowns=sampling_budget_breakdowns,
    )
