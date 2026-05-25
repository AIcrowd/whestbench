"""Tests for whestbench.load_dataset, iter_mlps, mlp_at, metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _bake_small(tmp_path: Path, *, split: str = "public") -> Path:
    from whestbench.dataset import create_dataset

    out = tmp_path / f"bake-{split}"
    create_dataset(n_mlps=3, n_samples=100, width=4, depth=2, seed=42, output_path=out, split=split)
    return out


def test_load_dataset_returns_datasets_dataset(tmp_path: Path):
    from datasets import Dataset

    from whestbench.dataset import load_dataset  # noqa: E402

    out = _bake_small(tmp_path)
    ds = load_dataset(out)
    assert isinstance(ds, Dataset)
    assert len(ds) == 3


def test_load_dataset_rejects_partial(tmp_path: Path):
    from whestbench.dataset import create_dataset, load_dataset
    from whestbench.dataset_io import InvalidDatasetError

    out = tmp_path / "partial"
    create_dataset(
        n_mlps=10, n_samples=50, width=4, depth=2, seed=1, output_path=out, mlp_range=(2, 5)
    )
    with pytest.raises(InvalidDatasetError, match="partial"):
        load_dataset(out)


def test_load_dataset_rejects_npz_file(tmp_path: Path):
    """An old schema-2.4 .npz file should produce a helpful error."""
    from whestbench.dataset import load_dataset
    from whestbench.dataset_io import InvalidDatasetError

    npz = tmp_path / "old.npz"
    np.savez(npz, foo=np.array([0]))
    with pytest.raises(InvalidDatasetError):
        load_dataset(npz)


def test_metadata_accessor_returns_parsed_json(tmp_path: Path):
    from whestbench.dataset import load_dataset, metadata

    out = _bake_small(tmp_path)
    ds = load_dataset(out)
    md = metadata(ds)
    assert md["schema_version"] == "3.0"
    assert md["seed"] == 42
    assert md["n_mlps"] == 3


def test_metadata_raises_for_bare_hf_load(tmp_path: Path):
    """datasets.load_dataset(...) directly should not get metadata."""
    from datasets import load_dataset as hf_load

    from whestbench.dataset import metadata

    out = _bake_small(tmp_path)
    ds = hf_load(str(out), split="public")
    with pytest.raises(KeyError):
        metadata(ds)


def test_iter_mlps_yields_validated_mlps(tmp_path: Path):
    from whestbench.dataset import iter_mlps, load_dataset
    from whestbench.domain import MLP

    out = _bake_small(tmp_path)
    ds = load_dataset(out)
    mlps = list(iter_mlps(ds))
    assert len(mlps) == 3
    for m in mlps:
        assert isinstance(m, MLP)
        assert m.width == 4
        assert m.depth == 2


def test_mlp_at_returns_indexed_mlp(tmp_path: Path):
    from whestbench.dataset import load_dataset, mlp_at

    out = _bake_small(tmp_path)
    ds = load_dataset(out)
    m0 = mlp_at(ds, 0)
    m2 = mlp_at(ds, 2)
    assert m0.seed != m2.seed
    assert m0.name == ds[0]["mlp_name"]
