"""make_contest_from_dataset must accept IterableDataset."""

from __future__ import annotations

import json

import flopscope.numpy as fnp
import numpy as np
import pytest
from datasets import Dataset, IterableDataset

from whestbench import metadata
from whestbench.scoring import ContestSpec, make_contest_from_dataset


def _fake_materialized_dataset(
    n: int, width: int = 4, depth: int = 2, *, bake_wall_time_s: float = 0.0
) -> Dataset:
    """Build a minimal Dataset matching whestbench's row shape.

    ``bake_wall_time_s`` mimics the torch bake, which stores the bake machine's
    wall clock in both ``wall_time_s`` and ``residual_wall_time_s``.
    """
    # Use the keys that _aggregate_budget_breakdowns indexes into directly.
    _zero_breakdown = {
        "flop_budget": 0,
        "flops_used": 0,
        "flops_remaining": 0,
        "wall_time_s": bake_wall_time_s,
        "flopscope_backend_time_s": 0.0,
        "flopscope_overhead_time_s": 0.0,
        "residual_wall_time_s": bake_wall_time_s,
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


def test_streaming_restore_keeps_bake_times_and_tags_source() -> None:
    """Streaming restore also keeps bake measurements verbatim and tags them."""
    n, width, depth = 3, 4, 2
    ds = _fake_materialized_dataset(n, width, depth, bake_wall_time_s=50.0)
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
    agg = contest.sampling_budget_breakdown
    assert agg is not None
    assert agg["residual_wall_time_s"] == pytest.approx(150.0)
    assert agg["wall_time_s"] == pytest.approx(150.0)
    assert agg["time_source"] == "bake"


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


@pytest.mark.parametrize("n", [1, 2])
def test_streaming_does_not_read_failing_row_after_requested_prefix(n: int) -> None:
    """An unavailable row outside the requested prefix must not fail the run."""
    rows = _fake_materialized_dataset(n).to_list()

    def generate():
        yield from rows
        raise OSError("The next, unrequested row is unavailable")

    spec = ContestSpec(width=4, depth=2, n_mlps=n, flop_budget=10_000_000, ground_truth_samples=10)
    contest = make_contest_from_dataset(
        spec, IterableDataset.from_generator(generate), n, seed_protocol_version="3.0"
    )
    assert len(contest.mlps) == n


@pytest.mark.parametrize("n", [1, 2])
def test_streaming_consumes_only_requested_rows_and_matches_materialized(n: int) -> None:
    ds = _fake_materialized_dataset(n + 1)
    rows = ds.to_list()
    consumed = []

    def generate():
        for row in rows:
            consumed.append(row["mlp_id"])
            yield row

    spec = ContestSpec(width=4, depth=2, n_mlps=n, flop_budget=10_000_000, ground_truth_samples=10)
    streamed = make_contest_from_dataset(
        spec, IterableDataset.from_generator(generate), n, seed_protocol_version="3.0"
    )
    materialized = make_contest_from_dataset(spec, ds, n)

    assert consumed == list(range(n))
    assert [mlp.seed for mlp in streamed.mlps] == [mlp.seed for mlp in materialized.mlps]
    assert [mlp.name for mlp in streamed.mlps] == [mlp.name for mlp in materialized.mlps]
    for actual, expected in zip(streamed.mlps, materialized.mlps):
        for actual_weight, expected_weight in zip(actual.weights, expected.weights):
            np.testing.assert_array_equal(np.asarray(actual_weight), np.asarray(expected_weight))
    for field in ("all_layer_targets", "final_targets"):
        for actual, expected in zip(getattr(streamed, field), getattr(materialized, field)):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert streamed.avg_variances == materialized.avg_variances
    assert streamed.sampling_budget_breakdown == materialized.sampling_budget_breakdown


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("row_width,row_depth", [(2, 2), (4, 1)])
@pytest.mark.parametrize("bad_index", [0, 1])
def test_dataset_geometry_must_match_spec(
    streaming: bool, row_width: int, row_depth: int, bad_index: int
) -> None:
    """Check every selected MLP, including a malformed row after a valid one."""
    rows = _fake_materialized_dataset(2).to_list()
    rows[bad_index] = _fake_materialized_dataset(1, row_width, row_depth)[0]
    ds = Dataset.from_list(rows)
    if streaming:
        ds = ds.to_iterable_dataset()
    spec = ContestSpec(width=4, depth=2, n_mlps=2, flop_budget=10_000_000, ground_truth_samples=10)

    with pytest.raises(ValueError, match=rf"Dataset MLP {bad_index} .*expected width=4, depth=2"):
        make_contest_from_dataset(spec, ds, 2, seed_protocol_version="3.0")


@pytest.mark.parametrize("streaming", [False, True])
def test_geometry_outside_requested_prefix_is_ignored(streaming: bool) -> None:
    rows = _fake_materialized_dataset(1).to_list()
    rows.extend(_fake_materialized_dataset(1, width=2, depth=1).to_list())
    ds = Dataset.from_list(rows)
    if streaming:
        ds = ds.to_iterable_dataset()
    spec = ContestSpec(width=4, depth=2, n_mlps=1, flop_budget=10_000_000, ground_truth_samples=10)

    contest = make_contest_from_dataset(spec, ds, 1, seed_protocol_version="3.0")

    assert len(contest.mlps) == 1
    assert contest.mlps[0].width == spec.width
    assert contest.mlps[0].depth == spec.depth
