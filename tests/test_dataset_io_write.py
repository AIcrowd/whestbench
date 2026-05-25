"""Tests for dataset_io.write_dataset_dir."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from whestbench.dataset_io import (
    DEFAULT_SPLIT,
    SCHEMA_FORMAT,
    SCHEMA_VERSION,
    make_features,
    write_dataset_dir,
)


def _make_tiny_dataset(n_mlps: int = 2, width: int = 4, depth: int = 2) -> Dataset:
    """Build a tiny Dataset matching the schema, for write/read tests."""
    rng = np.random.default_rng(0)
    return Dataset.from_dict(
        {
            "mlp_id": list(range(n_mlps)),
            "mlp_name": [f"slug-{i}" for i in range(n_mlps)],
            "mlp_seed": [int(rng.integers(0, 2**31)) for _ in range(n_mlps)],
            "weights": rng.standard_normal((n_mlps, depth, width, width)).astype("float32"),
            "all_layer_means": rng.standard_normal((n_mlps, depth, width)).astype("float32"),
            "final_means": rng.standard_normal((n_mlps, width)).astype("float32"),
            "avg_variance": rng.random(n_mlps).astype("float64"),
            "sampling_budget_breakdown": ["{}"] * n_mlps,
        },
        features=make_features(width=width, depth=depth),
    )


def _make_metadata(**overrides):
    base = {
        "schema_version": SCHEMA_VERSION,
        "format": SCHEMA_FORMAT,
        "backend": "flopscope",
        "seed_protocol": {
            "name": "whestbench_seedsequence_hierarchy",
            "version": "2.0",
            "seeded": True,
        },
        "created_at_utc": "2026-05-25T00:00:00+00:00",
        "seed": 42,
        "n_mlps": 2,
        "n_samples": 1000,
        "width": 4,
        "depth": 2,
        "hardware": {"cpu_brand": "test"},
    }
    base.update(overrides)
    return base


def test_write_creates_three_files(tmp_path: Path):
    ds = _make_tiny_dataset()
    metadata = _make_metadata()
    out = tmp_path / "ds"
    write_dataset_dir(ds, output_dir=out, split=DEFAULT_SPLIT, metadata=metadata)

    assert (out / "data" / "public-00000-of-00001.parquet").is_file()
    assert (out / "metadata.json").is_file()
    assert (out / "README.md").is_file()


def test_write_metadata_json_round_trips(tmp_path: Path):
    ds = _make_tiny_dataset()
    metadata = _make_metadata(seed=99)
    out = tmp_path / "ds"
    write_dataset_dir(ds, output_dir=out, split=DEFAULT_SPLIT, metadata=metadata)

    loaded = json.loads((out / "metadata.json").read_text())
    assert loaded == metadata


def test_write_refuses_existing_directory(tmp_path: Path):
    ds = _make_tiny_dataset()
    metadata = _make_metadata()
    out = tmp_path / "ds"
    out.mkdir()

    with pytest.raises(FileExistsError):
        write_dataset_dir(ds, output_dir=out, split=DEFAULT_SPLIT, metadata=metadata)


def test_write_with_holdout_split_name(tmp_path: Path):
    ds = _make_tiny_dataset()
    metadata = _make_metadata()
    out = tmp_path / "ds"
    write_dataset_dir(ds, output_dir=out, split="holdout", metadata=metadata)
    assert (out / "data" / "holdout-00000-of-00001.parquet").is_file()
