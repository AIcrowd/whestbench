"""Create, save, and load whestbench evaluation datasets (schema 3.0)."""

from __future__ import annotations

import dataclasses
import json
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset

from .dataset_io import (
    DEFAULT_SPLIT,
    SCHEMA_FORMAT,
    SCHEMA_VERSION,
    SEED_PROTOCOL_NAME,
    SEED_PROTOCOL_VERSION,
    make_features,
    write_dataset_dir,
)
from .domain import MLP
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint
from .naming import assign_unique_names
from .scoring import _normalize_sampling_budget_breakdown
from .simulation import sample_layer_statistics, sample_layer_statistics_chunk_count

# Metadata side-channel: associates a Dataset with its metadata.json contents
# without mutating the Dataset object itself.
_METADATA_BY_DS: "weakref.WeakKeyDictionary[Dataset, Dict[str, Any]]" = weakref.WeakKeyDictionary()


def _resolve_mlp_range(n_mlps: int, mlp_range: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if mlp_range is None:
        return (0, n_mlps)
    start, end = mlp_range
    if not (0 <= start < end <= n_mlps):
        raise ValueError(
            f"mlp_range {mlp_range!r} invalid for n_mlps={n_mlps}; need 0 <= start < end <= n_mlps."
        )
    return (start, end)


def create_dataset(
    *,
    n_mlps: int,
    n_samples: int,
    width: int,
    depth: int,
    seed: Optional[int] = None,
    output_path: "Path | str",
    split: str = DEFAULT_SPLIT,
    mlp_range: Optional[Tuple[int, int]] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Path:
    """Generate MLPs, compute ground-truth, and write a schema-3.0 dataset directory.

    Output is a directory with data/<split>-00000-of-00001.parquet, metadata.json,
    README.md. Raises FileExistsError if output_path already exists.

    If `mlp_range=(start, end)` is set, only MLPs in [start, end) are generated.
    Output metadata is marked is_partial=true. Run merge_datasets to combine.

    Bit-equivalent property: a worker baking slice [a, b) of a logical dataset of
    size N produces the same rows as the corresponding slice of a single-host bake
    of size N.
    """
    output_path = Path(output_path)
    start, end = _resolve_mlp_range(n_mlps, mlp_range)

    seed_sequence = (
        fnp.random.SeedSequence() if seed is None else fnp.random.SeedSequence(int(seed))
    )
    stream_seed = seed_sequence.spawn(3 * n_mlps)

    # Phase 1: generate MLPs in the slice
    mlps: List[MLP] = []
    for slice_idx, i in enumerate(range(start, end)):
        weight_stream = fnp.random.default_rng(stream_seed[3 * i])
        estimator_seed_i = int(stream_seed[3 * i + 2].generate_state(1)[0])
        mlps.append(sample_mlp(width, depth, weight_stream, seed=estimator_seed_i))
        if progress is not None:
            progress({"phase": "generating", "completed": slice_idx + 1, "total": end - start})

    # Names: derived from ALL logical seeds, then sliced. Guarantees slice's
    # names match the corresponding slice of a single-host bake.
    all_logical_seeds = [int(stream_seed[3 * i + 2].generate_state(1)[0]) for i in range(n_mlps)]
    all_names = assign_unique_names(all_logical_seeds)
    slice_names = all_names[start:end]
    mlps = [dataclasses.replace(m, name=n) for m, n in zip(mlps, slice_names)]

    weights_array = np.stack([np.stack(mlp.weights) for mlp in mlps]).astype(np.float32)

    # Phase 2: ground-truth sampling
    all_means_list: List[fnp.ndarray] = []
    final_means_list: List[fnp.ndarray] = []
    avg_variances: List[float] = []
    sampling_budget_breakdowns: List[Dict[str, Any]] = []
    chunks_per_mlp = sample_layer_statistics_chunk_count(width, n_samples)
    total_sampling_chunks = (end - start) * chunks_per_mlp

    for slice_idx, i in enumerate(range(start, end)):
        sample_stream = fnp.random.default_rng(stream_seed[3 * i + 1])
        mlp = mlps[slice_idx]

        def _on_chunk(
            event,
            *,
            mlp_index=slice_idx + 1,
            name=mlp.name,
            chunk_offset=slice_idx * chunks_per_mlp,
        ):
            if progress is None:
                return
            local_completed = int(event.get("completed", 0))
            progress(
                {
                    "phase": "sampling",
                    "completed": chunk_offset + local_completed,
                    "total": total_sampling_chunks,
                    "mlp_index": mlp_index,
                    "mlp_name": name,
                    "n_mlps": end - start,
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
                        progress=_on_chunk if progress is not None else None,
                    )
        normalized = _normalize_sampling_budget_breakdown(
            sampling_budget.summary_dict(by_namespace=True)
        )
        if normalized is not None:
            sampling_budget_breakdowns.append(normalized)
        all_means_list.append(fnp.asarray(all_means, dtype=fnp.float32))  # pyright: ignore[reportPossiblyUnboundVariable]
        final_means_list.append(fnp.asarray(final_mean, dtype=fnp.float32))  # pyright: ignore[reportPossiblyUnboundVariable]
        avg_variances.append(avg_var)  # pyright: ignore[reportPossiblyUnboundVariable]

    all_layer_means = np.stack(all_means_list).astype(np.float32)
    final_means = np.stack(final_means_list).astype(np.float32)

    ds = Dataset.from_dict(
        {
            "mlp_id": list(range(start, end)),
            "mlp_name": [m.name for m in mlps],
            "mlp_seed": [int(m.seed) for m in mlps],
            "weights": weights_array,
            "all_layer_means": all_layer_means,
            "final_means": final_means,
            "avg_variance": avg_variances,
            "sampling_budget_breakdown": [json.dumps(b) for b in sampling_budget_breakdowns],
        },
        features=make_features(width=width, depth=depth),
    )

    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "format": SCHEMA_FORMAT,
        "backend": "flopscope",
        "seed_protocol": {
            "name": SEED_PROTOCOL_NAME,
            "version": SEED_PROTOCOL_VERSION,
            "seeded": seed is not None,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed_sequence.entropy),
        "n_mlps": end - start,
        "n_samples": n_samples,
        "width": width,
        "depth": depth,
        "hardware": collect_hardware_fingerprint(),
    }

    is_partial = (start, end) != (0, n_mlps)
    if is_partial:
        metadata["is_partial"] = True
        metadata["mlp_range"] = [start, end]
        metadata["total_n_mlps"] = n_mlps

    write_dataset_dir(ds, output_dir=output_path, split=split, metadata=metadata)
    return output_path
