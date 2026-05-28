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


def test_cpu_bake_v3_is_internally_consistent(tmp_path: Path):
    """Bake with explicit mlp_seeds; verify weights/means are deterministic.

    The v3 protocol changes the seed derivation vs the legacy 2.4 npz format,
    so bit-equivalence between the two is intentionally NOT tested here.
    Instead we verify v3 internal consistency: two bakes with the same mlp_seeds
    produce bit-identical output.

    Timing fields (wall_time_s, flopscope_*_time_s, residual_wall_time_s) are
    wall-clock measurements and are not deterministic between runs; only the
    deterministic fields are compared.
    """
    from datasets import load_dataset as hf_load_dataset
    from whestbench.dataset import create_dataset
    from whestbench.dataset_io import DEFAULT_SPLIT

    mlp_seeds = [42000, 42001, 42002]
    common: dict[str, Any] = dict(n_mlps=3, n_samples=100, width=4, depth=2, mlp_seeds=mlp_seeds)

    dir_a = tmp_path / "bake_a"
    create_dataset(output_path=dir_a, **common)
    dir_b = tmp_path / "bake_b"
    create_dataset(output_path=dir_b, **common)

    ds_a = hf_load_dataset(str(dir_a), split=DEFAULT_SPLIT)
    ds_b = hf_load_dataset(str(dir_b), split=DEFAULT_SPLIT)

    np.testing.assert_array_equal(np.array(ds_a["weights"]), np.array(ds_b["weights"]))
    np.testing.assert_array_equal(
        np.array(ds_a["all_layer_means"]), np.array(ds_b["all_layer_means"])
    )
    np.testing.assert_array_equal(np.array(ds_a["final_means"]), np.array(ds_b["final_means"]))
    np.testing.assert_array_equal(
        np.array(ds_a["avg_variance"]).astype("float64"),
        np.array(ds_b["avg_variance"]).astype("float64"),
    )
    # Under v3, parquet mlp_seed IS the input seed.
    assert ds_a["mlp_seed"] == mlp_seeds
    assert ds_a["mlp_seed"] == ds_b["mlp_seed"]
    assert ds_a["mlp_name"] == ds_b["mlp_name"]
    for i, (bd_a_json, bd_b_json) in enumerate(
        zip(ds_a["sampling_budget_breakdown"], ds_b["sampling_budget_breakdown"])
    ):
        bd_a = _strip_timing(json.loads(bd_a_json))
        bd_b = _strip_timing(json.loads(bd_b_json))
        assert bd_a == bd_b, f"breakdown[{i}] deterministic fields differ"


def test_cpu_bake_produces_three_file_layout(tmp_path: Path):
    from whestbench.dataset import create_dataset

    out = tmp_path / "ds"
    create_dataset(
        n_mlps=2, n_samples=50, width=4, depth=2, mlp_seeds=[1000, 1001], output_path=out
    )
    assert (out / "data" / "public-00000-of-00001.parquet").is_file()
    assert (out / "metadata.json").is_file()
    assert (out / "README.md").is_file()


def test_cpu_bake_metadata_has_schema_3_0(tmp_path: Path):
    from whestbench.dataset import create_dataset

    out = tmp_path / "ds"
    create_dataset(
        n_mlps=2,
        n_samples=50,
        width=4,
        depth=2,
        mlp_seeds=[1000, 1001],
        output_path=out,
        split="mini",
        config="default",
    )
    md = json.loads((out / "metadata.json").read_text())
    assert md["schema_version"] == "3.0"
    assert md["format"] == "hf-datasets-parquet"
    assert md["backend"] == "flopscope"
    assert md["split"] == "mini"
    assert md["config"] == "default"


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
        mlp_seeds=list(range(10)),
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
