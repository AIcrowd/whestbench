"""Torch-path round-trip: parquet output matches CPU bake within ~3e-5 (statistical)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_torch_bake_metadata_has_schema_3_0(tmp_path: Path):
    from whestbench.dataset_torch import create_dataset_torch

    out = tmp_path / "ds"
    create_dataset_torch(
        n_mlps=2,
        n_samples=10_000,
        width=4,
        depth=2,
        mlp_seeds=[7, 8],
        output_path=out,
        device="cpu",
    )
    md = json.loads((out / "metadata.json").read_text())
    assert md["schema_version"] == "3.0"
    assert md["format"] == "hf-datasets-parquet"
    assert md["backend"] == "torch"
    assert md["device"] == "cpu"


def test_torch_bake_three_file_layout(tmp_path: Path):
    from whestbench.dataset_torch import create_dataset_torch

    out = tmp_path / "ds"
    create_dataset_torch(
        n_mlps=2,
        n_samples=10_000,
        width=4,
        depth=2,
        mlp_seeds=[7, 8],
        output_path=out,
        device="cpu",
    )
    assert (out / "data" / "public-00000-of-00001.parquet").is_file()
    assert (out / "metadata.json").is_file()


def test_torch_cpu_means_within_tolerance_of_flopscope_path(tmp_path: Path):
    """At the same mlp_seeds, torch CPU mode statistically matches flopscope CPU."""
    from datasets import load_dataset as hf_load_dataset

    from whestbench.dataset import create_dataset
    from whestbench.dataset_torch import create_dataset_torch

    # dict[str, Any] so pyright doesn't widen the (homogeneous-int) literal
    # type onto kwargs like `progress`, `split`, `mlp_range` when **-spread.
    common: dict[str, Any] = dict(n_mlps=2, n_samples=50_000, width=4, depth=2, mlp_seeds=[42, 43])
    cpu_dir = tmp_path / "cpu"
    torch_dir = tmp_path / "torch"
    create_dataset(output_path=cpu_dir, **common)
    create_dataset_torch(output_path=torch_dir, device="cpu", **common)

    ds_cpu = hf_load_dataset(str(cpu_dir), split="public")
    ds_torch = hf_load_dataset(str(torch_dir), split="public")

    np.testing.assert_allclose(
        np.array(ds_torch["all_layer_means"]),
        np.array(ds_cpu["all_layer_means"]),
        rtol=0,
        atol=3e-2,  # broader tol at small N; 3e-5 holds at N=1e9
    )
    assert ds_cpu["mlp_seed"] == ds_torch["mlp_seed"]
    assert ds_cpu["mlp_name"] == ds_torch["mlp_name"]


def test_torch_bake_supports_mlp_range(tmp_path: Path):
    """Slicing on the torch path mirrors CPU behavior."""
    from datasets import load_dataset as hf_load_dataset

    from whestbench.dataset_torch import create_dataset_torch

    out = tmp_path / "partial"
    create_dataset_torch(
        n_mlps=8,
        n_samples=10_000,
        width=4,
        depth=2,
        mlp_seeds=[1, 2, 3, 4, 5, 6, 7, 8],
        output_path=out,
        mlp_range=(2, 5),
        device="cpu",
    )
    ds = hf_load_dataset(str(out), split="public")
    assert len(ds) == 3
    assert ds["mlp_id"] == [2, 3, 4]
    md = json.loads((out / "metadata.json").read_text())
    assert md["is_partial"] is True
    assert md["mlp_range"] == [2, 5]
