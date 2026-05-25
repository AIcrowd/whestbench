"""Frozen schema 2.4 npz writer/reader, kept ONLY for round-trip equivalence
tests during the 2.4→3.0 migration. DO NOT use in production code.

This is a verbatim copy of the schema 2.4 logic from src/whestbench/dataset.py
at the time of the schema break, with public function names prefixed by `legacy_`.
Delete this file after schema 3.0 has shipped to one minor release.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from whestbench.domain import MLP
from whestbench.generation import sample_mlp
from whestbench.hardware import collect_hardware_fingerprint
from whestbench.naming import assign_unique_names
from whestbench.scoring import _normalize_sampling_budget_breakdown
from whestbench.simulation import sample_layer_statistics

LEGACY_SCHEMA_VERSION = "2.4"
LEGACY_SEED_PROTOCOL_NAME = "whestbench_seedsequence_hierarchy"
LEGACY_SEED_PROTOCOL_VERSION = "2.0"


def legacy_create_dataset_npz(
    *,
    n_mlps: int,
    n_samples: int,
    width: int,
    depth: int,
    seed: Optional[int] = None,
    output_path: "Path | str",
) -> Path:
    """Schema 2.4 npz writer — for migration round-trip tests only."""
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

    mlp_names_list = assign_unique_names([m.seed for m in mlps])
    mlps = [dataclasses.replace(m, name=n) for m, n in zip(mlps, mlp_names_list)]

    weights_array = np.stack([np.stack(mlp.weights) for mlp in mlps]).astype(np.float32)

    all_means_list: List[fnp.ndarray] = []
    final_means_list: List[fnp.ndarray] = []
    avg_variances: List[float] = []
    sampling_budget_breakdowns: List[Dict[str, Any]] = []
    for i, mlp in enumerate(mlps):
        sample_stream = fnp.random.default_rng(stream_seed[3 * i + 1])
        with flops.BudgetContext(flop_budget=int(1e15), quiet=True) as sampling_budget:
            with flops.namespace("sampling"):
                with flops.namespace("sample_layer_statistics"):
                    all_means, final_mean, avg_var = sample_layer_statistics(
                        mlp, n_samples, rng=sample_stream
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

    metadata: Dict[str, Any] = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "backend": "flopscope",
        "seed_protocol": {
            "name": LEGACY_SEED_PROTOCOL_NAME,
            "version": LEGACY_SEED_PROTOCOL_VERSION,
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


def legacy_load_npz_arrays(path: "Path | str") -> Dict[str, Any]:
    """Return all arrays + parsed metadata from a schema 2.4 .npz file."""
    data = np.load(path, allow_pickle=False)
    return {
        "metadata": json.loads(str(data["metadata"])),
        "weights": data["weights"].astype(np.float32),
        "all_layer_means": data["all_layer_means"].astype(np.float32),
        "final_means": data["final_means"].astype(np.float32),
        "avg_variances": data["avg_variances"].astype(np.float64).tolist(),
        "sampling_budget_breakdowns": json.loads(str(data["sampling_budget_breakdowns"])),
        "mlp_seeds": data["mlp_seeds"].astype(np.int64).tolist(),
        "mlp_names": [str(s) for s in data["mlp_names"].tolist()],
    }
