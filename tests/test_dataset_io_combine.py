"""Tests for dataset_io.combine_split_datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _bake_single_split(
    tmp_path: Path,
    name: str,
    *,
    split: str = "public",
    n_mlps: int = 2,
    seed: int = 42,
    width: int = 8,
    depth: int = 2,
    n_samples: int = 100,
):
    from whestbench.dataset import create_dataset

    out = tmp_path / name
    create_dataset(
        n_mlps=n_mlps,
        n_samples=n_samples,
        width=width,
        depth=depth,
        seed=seed,
        output_path=out,
        split=split,
    )
    return out


def test_combine_splits_roundtrip(tmp_path: Path):
    from whestbench.dataset_io import combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub-bake", split="public", seed=42)
    hold = _bake_single_split(tmp_path, "hold-bake", split="holdout", seed=99)
    out = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=out)

    assert (out / "data" / "public-00000-of-00001.parquet").is_file()
    assert (out / "data" / "holdout-00000-of-00001.parquet").is_file()
    assert (out / "metadata.json").is_file()
    assert (out / "README.md").is_file()

    md = json.loads((out / "metadata.json").read_text())
    assert md["schema_version"] == "3.0"
    assert "n_mlps" not in md  # promoted to per-split
    assert "seed" not in md
    assert md["width"] == 8
    assert md["depth"] == 2
    assert md["n_samples"] == 100
    assert md["backend"] == "flopscope"
    assert md["splits"]["public"]["n_mlps"] == 2
    assert md["splits"]["public"]["seed"] == 42
    assert md["splits"]["holdout"]["n_mlps"] == 2
    assert md["splits"]["holdout"]["seed"] == 99


def test_combine_splits_returns_output_path(tmp_path: Path):
    from whestbench.dataset_io import combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public")
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=2)
    out = tmp_path / "combined"
    result = combine_split_datasets([pub, hold], output_dir=out)
    assert result == out


def test_combine_splits_rejects_existing_output(tmp_path: Path):
    from whestbench.dataset_io import combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub")
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=2)
    out = tmp_path / "combined"
    out.mkdir()
    with pytest.raises(FileExistsError, match=r"already exists"):
        combine_split_datasets([pub, hold], output_dir=out)


def test_combine_splits_rejects_empty_inputs(tmp_path: Path):
    from whestbench.dataset_io import MergeIncompatibleError, combine_split_datasets

    with pytest.raises(MergeIncompatibleError, match=r"at least one"):
        combine_split_datasets([], output_dir=tmp_path / "x")


def test_combine_splits_accepts_single_input(tmp_path: Path):
    """One input is a legal degenerate case (single-split -> multi-split layout)."""
    from whestbench.dataset_io import combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public")
    out = tmp_path / "combined"
    combine_split_datasets([pub], output_dir=out)
    md = json.loads((out / "metadata.json").read_text())
    assert set(md["splits"].keys()) == {"public"}


def test_combine_splits_rejects_partial_input(tmp_path: Path):
    from whestbench.dataset import create_dataset
    from whestbench.dataset_io import MergeIncompatibleError, combine_split_datasets

    partial = tmp_path / "partial"
    create_dataset(
        n_mlps=8,
        n_samples=100,
        width=4,
        depth=2,
        seed=1,
        output_path=partial,
        mlp_range=(0, 4),
    )
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=2)
    with pytest.raises(MergeIncompatibleError, match=r"partial"):
        combine_split_datasets([partial, hold], output_dir=tmp_path / "x")


def test_combine_splits_rejects_duplicate_split_names(tmp_path: Path):
    from whestbench.dataset_io import MergeIncompatibleError, combine_split_datasets

    pub1 = _bake_single_split(tmp_path, "pub1", split="public", seed=1)
    pub2 = _bake_single_split(tmp_path, "pub2", split="public", seed=2)
    with pytest.raises(MergeIncompatibleError, match=r"duplicate"):
        combine_split_datasets([pub1, pub2], output_dir=tmp_path / "x")


def test_combine_splits_rejects_mismatched_width(tmp_path: Path):
    from whestbench.dataset_io import MergeIncompatibleError, combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public", width=8)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", width=16, seed=2)
    with pytest.raises(MergeIncompatibleError, match=r"width"):
        combine_split_datasets([pub, hold], output_dir=tmp_path / "x")


def test_combine_splits_rejects_mismatched_depth(tmp_path: Path):
    from whestbench.dataset_io import MergeIncompatibleError, combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public", depth=2)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", depth=3, seed=2)
    with pytest.raises(MergeIncompatibleError, match=r"depth"):
        combine_split_datasets([pub, hold], output_dir=tmp_path / "x")


def test_combine_splits_rejects_mismatched_n_samples(tmp_path: Path):
    from whestbench.dataset_io import MergeIncompatibleError, combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public", n_samples=100)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", n_samples=200, seed=2)
    with pytest.raises(MergeIncompatibleError, match=r"n_samples"):
        combine_split_datasets([pub, hold], output_dir=tmp_path / "x")


def test_combine_splits_validates_combined_metadata_against_schema(tmp_path: Path):
    """Output metadata must pass validate_metadata in multi-split mode."""
    from whestbench.dataset_io import (
        combine_split_datasets,
        read_metadata,
        validate_metadata,
    )

    pub = _bake_single_split(tmp_path, "pub", split="public")
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=2)
    out = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=out)
    md = read_metadata(out)
    validate_metadata(md)  # must not raise


def test_combine_split_datasets_is_re_exported():
    import whestbench

    assert hasattr(whestbench, "combine_split_datasets")
    assert "combine_split_datasets" in whestbench.__all__


def _patch_metadata(dir_path: Path, **patches):
    """Mutate a dataset directory's metadata.json with the given key/value patches."""
    md_path = dir_path / "metadata.json"
    md = json.loads(md_path.read_text())
    md.update(patches)
    md_path.write_text(json.dumps(md, indent=2))


def test_combine_splits_rejects_mismatched_backend(tmp_path: Path):
    from whestbench.dataset_io import MergeIncompatibleError, combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public")
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=2)
    _patch_metadata(hold, backend="torch")  # bake produced "flopscope"
    with pytest.raises(MergeIncompatibleError, match=r"backend"):
        combine_split_datasets([pub, hold], output_dir=tmp_path / "x")


def test_combine_splits_rejects_mismatched_schema_version(tmp_path: Path):
    # validate_metadata (called before the cross-input invariant check) also
    # validates schema_version, so patching to "2.9" may raise InvalidDatasetError
    # rather than MergeIncompatibleError.  Accept either.
    from whestbench.dataset_io import (
        InvalidDatasetError,
        MergeIncompatibleError,
        combine_split_datasets,
    )

    pub = _bake_single_split(tmp_path, "pub", split="public")
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=2)
    _patch_metadata(hold, schema_version="2.9")
    with pytest.raises((MergeIncompatibleError, InvalidDatasetError), match=r"schema_version"):
        combine_split_datasets([pub, hold], output_dir=tmp_path / "x")


def test_combine_splits_rejects_mismatched_format(tmp_path: Path):
    from whestbench.dataset_io import MergeIncompatibleError, combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public")
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=2)
    _patch_metadata(hold, format="some-other-format")
    with pytest.raises(MergeIncompatibleError, match=r"format"):
        combine_split_datasets([pub, hold], output_dir=tmp_path / "x")


def test_combine_splits_rejects_mismatched_seed_protocol(tmp_path: Path):
    # validate_metadata checks seed_protocol.version before the cross-input
    # invariant check fires, so a patched seed_protocol may raise
    # InvalidDatasetError instead of MergeIncompatibleError.  Accept either.
    from whestbench.dataset_io import (
        InvalidDatasetError,
        MergeIncompatibleError,
        combine_split_datasets,
    )

    pub = _bake_single_split(tmp_path, "pub", split="public")
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=2)
    _patch_metadata(
        hold,
        seed_protocol={"name": "whestbench_seedsequence_hierarchy", "version": "1.0"},
    )
    with pytest.raises((MergeIncompatibleError, InvalidDatasetError), match=r"seed_protocol"):
        combine_split_datasets([pub, hold], output_dir=tmp_path / "x")
