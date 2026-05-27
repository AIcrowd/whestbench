"""make_contest_from_dataset must accept IterableDataset."""

from __future__ import annotations

import json

import flopscope.numpy as fnp
import numpy as np
import pytest
from datasets import Dataset

from whestbench import metadata
from whestbench.scoring import ContestSpec, make_contest_from_dataset


def _fake_materialized_dataset(n: int, width: int = 4, depth: int = 2) -> Dataset:
    """Build a minimal Dataset matching whestbench's row shape."""
    # Use the keys that _aggregate_budget_breakdowns indexes into directly.
    _zero_breakdown = {
        "flop_budget": 0,
        "flops_used": 0,
        "flops_remaining": 0,
        "wall_time_s": 0.0,
        "flopscope_backend_time_s": 0.0,
        "flopscope_overhead_time_s": 0.0,
        "residual_wall_time_s": 0.0,
        "by_namespace": {},
    }
    rows = []
    for i in range(n):
        rows.append(
            {
                "mlp_id": i,
                "mlp_name": f"name-{i}",
                "mlp_seed": 1000 + i,
                "weights": np.zeros((depth, width, width), dtype=np.float32).tolist(),
                "all_layer_means": np.zeros((depth, width), dtype=np.float32).tolist(),
                "final_means": np.zeros(width, dtype=np.float32).tolist(),
                "avg_variance": 0.5,
                "sampling_budget_breakdown": json.dumps(_zero_breakdown),
            }
        )
    ds = Dataset.from_list(rows)
    # Attach metadata via the side-channel, mimicking what load_dataset() does.
    from whestbench.dataset import _METADATA_BY_DS

    _METADATA_BY_DS[ds] = {
        "schema_version": "3.0",
        "format": "parquet",
        "backend": "flopscope",
        "n_mlps": n,
        "n_samples": 10,
        "width": width,
        "depth": depth,
        "seed_protocol": {"name": "whestbench_explicit_per_mlp_seeds", "version": "3.0"},
    }
    return ds


def test_make_contest_accepts_iterable_dataset() -> None:
    n, width, depth = 5, 4, 2
    ds = _fake_materialized_dataset(n, width, depth)
    spec = ContestSpec(
        width=width,
        depth=depth,
        n_mlps=n,
        flop_budget=10_000_000,
        ground_truth_samples=10,
        seed=0,
        wall_time_limit_s=None,
        residual_wall_time_limit_s=None,
    )

    iter_ds = ds.to_iterable_dataset()
    from whestbench.dataset import _METADATA_BY_DS

    _METADATA_BY_DS[iter_ds] = metadata(ds)

    contest = make_contest_from_dataset(spec, iter_ds, n)
    assert len(contest.mlps) == n
    assert len(contest.all_layer_targets) == n
    assert len(contest.final_targets) == n
    assert len(contest.avg_variances) == n
    # Spot-check shapes — every entry should match (depth, width) / (width,).
    for i in range(n):
        arr = fnp.asarray(contest.all_layer_targets[i])
        assert tuple(arr.shape) == (depth, width)
        final = fnp.asarray(contest.final_targets[i])
        assert tuple(final.shape) == (width,)


def test_make_contest_streaming_too_few_rows_raises() -> None:
    n, width, depth = 5, 4, 2
    ds = _fake_materialized_dataset(n, width, depth)
    spec = ContestSpec(
        width=width,
        depth=depth,
        n_mlps=10,
        flop_budget=10_000_000,
        ground_truth_samples=10,
        seed=0,
        wall_time_limit_s=None,
        residual_wall_time_limit_s=None,
    )

    iter_ds = ds.to_iterable_dataset()
    from whestbench.dataset import _METADATA_BY_DS

    _METADATA_BY_DS[iter_ds] = metadata(ds)

    with pytest.raises(ValueError, match="yielded only 5 MLPs"):
        make_contest_from_dataset(spec, iter_ds, 10)
