"""CPU-path round-trip equivalence: legacy 2.4 npz vs schema 3.0 Parquet+sidecar."""

# ruff: noqa: I001
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Adjust import path for the test-only legacy module; must come before the import.
sys.path.insert(0, "tests")
from _legacy_npz import legacy_create_dataset_npz, legacy_load_npz_arrays  # noqa: E402


_TIMING_KEYS = frozenset(
    [
        "wall_time_s",
        "flopscope_backend_time_s",
        "flopscope_overhead_time_s",
        "residual_wall_time_s",
    ]
)


def _strip_timing(d: dict) -> dict:
    """Recursively drop wall-clock timing keys that are non-deterministic."""
    out = {}
    for k, v in d.items():
        if k in _TIMING_KEYS:
            continue
        if isinstance(v, dict):
            out[k] = _strip_timing(v)
        else:
            out[k] = v
    return out


def test_cpu_bake_bit_equivalent_old_vs_new(tmp_path: Path):
    """Bake the same (seed, n_mlps, ...) via both paths; assert bit-identical.

    Timing fields (wall_time_s, flopscope_*_time_s, residual_wall_time_s) are
    wall-clock measurements and are not deterministic between runs; only the
    deterministic fields (flops_used, flop_cost, calls, ...) are compared.
    """
    from datasets import load_dataset as hf_load_dataset
    from whestbench.dataset import create_dataset
    from whestbench.dataset_io import DEFAULT_SPLIT

    # dict[str, Any] so pyright doesn't widen the (homogeneous-int) literal
    # type onto kwargs like `progress`, `split`, `mlp_range` when **-spread.
    common: dict[str, Any] = dict(n_mlps=3, n_samples=100, width=4, depth=2, seed=42)

    old_npz = tmp_path / "old.npz"
    legacy_create_dataset_npz(output_path=old_npz, **common)
    old = legacy_load_npz_arrays(old_npz)

    new_dir = tmp_path / "new"
    create_dataset(output_path=new_dir, **common)
    ds = hf_load_dataset(str(new_dir), split=DEFAULT_SPLIT)

    np.testing.assert_array_equal(np.array(ds["weights"]), old["weights"])
    np.testing.assert_array_equal(np.array(ds["all_layer_means"]), old["all_layer_means"])
    np.testing.assert_array_equal(np.array(ds["final_means"]), old["final_means"])
    np.testing.assert_array_equal(
        np.array(ds["avg_variance"]).astype("float64"),
        np.array(old["avg_variances"], dtype="float64"),
    )
    assert ds["mlp_seed"] == old["mlp_seeds"]
    assert ds["mlp_name"] == old["mlp_names"]
    for i, breakdown_json in enumerate(ds["sampling_budget_breakdown"]):
        new_bd = _strip_timing(json.loads(breakdown_json))
        old_bd = _strip_timing(old["sampling_budget_breakdowns"][i])
        assert new_bd == old_bd, f"breakdown[{i}] deterministic fields differ"


def test_cpu_bake_produces_three_file_layout(tmp_path: Path):
    from whestbench.dataset import create_dataset

    out = tmp_path / "ds"
    create_dataset(n_mlps=2, n_samples=50, width=4, depth=2, seed=1, output_path=out)
    assert (out / "data" / "public-00000-of-00001.parquet").is_file()
    assert (out / "metadata.json").is_file()
    assert (out / "README.md").is_file()


def test_cpu_bake_metadata_has_schema_3_0(tmp_path: Path):
    from whestbench.dataset import create_dataset

    out = tmp_path / "ds"
    create_dataset(n_mlps=2, n_samples=50, width=4, depth=2, seed=1, output_path=out)
    md = json.loads((out / "metadata.json").read_text())
    assert md["schema_version"] == "3.0"
    assert md["format"] == "hf-datasets-parquet"
    assert md["backend"] == "flopscope"


def test_cpu_bake_supports_mlp_range(tmp_path: Path):
    """Slicing: a partial bake covers only the requested mlp_range."""
    from datasets import load_dataset as hf_load_dataset
    from whestbench.dataset import create_dataset

    out = tmp_path / "partial"
    create_dataset(
        n_mlps=10,
        n_samples=50,
        width=4,
        depth=2,
        seed=1,
        output_path=out,
        mlp_range=(3, 7),
    )
    ds = hf_load_dataset(str(out), split="public")
    assert len(ds) == 4
    assert ds["mlp_id"] == [3, 4, 5, 6]

    md = json.loads((out / "metadata.json").read_text())
    assert md["is_partial"] is True
    assert md["mlp_range"] == [3, 7]
    assert md["total_n_mlps"] == 10
    assert md["n_mlps"] == 4
