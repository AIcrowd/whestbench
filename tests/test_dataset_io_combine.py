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
    config: str = "default",
    n_mlps: int = 2,
    seed: int = 42,  # kept for call-site compatibility; used to derive mlp_seeds
    width: int = 8,
    depth: int = 2,
    n_samples: int = 100,
):
    from whestbench.dataset import create_dataset

    mlp_seeds = [seed * 1000 + i for i in range(n_mlps)]
    out = tmp_path / name
    create_dataset(
        n_mlps=n_mlps,
        n_samples=n_samples,
        width=width,
        depth=depth,
        mlp_seeds=mlp_seeds,
        output_path=out,
        split=split,
        config=config,
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
    assert "seed" not in md  # no top-level seed under v3
    assert md["width"] == 8
    assert md["depth"] == 2
    assert md["n_samples"] == 100
    assert md["backend"] == "flopscope"
    assert md["splits"]["public"]["n_mlps"] == 2
    # Under v3, per-split `seed` is not stored — seeds live in the parquet column.
    assert "seed" not in md["splits"]["public"]
    assert md["splits"]["holdout"]["n_mlps"] == 2
    assert "seed" not in md["splits"]["holdout"]


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
        mlp_seeds=[1000 + i for i in range(8)],
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


# -----------------------------------------------------------------------------
# default_split
# -----------------------------------------------------------------------------


def test_combine_splits_writes_default_split_when_given(tmp_path: Path):
    """A valid default_split is recorded at the top level of metadata.json."""
    from whestbench.dataset_io import combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public", seed=42)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=99)
    out = tmp_path / "combined"
    combine_split_datasets(
        [pub, hold],
        output_dir=out,
        default_split="public",
    )

    md = json.loads((out / "metadata.json").read_text())
    assert md["default_split"] == "public"


def test_combine_splits_preserves_declared_configs_and_auto_default_split(tmp_path: Path):
    from whestbench.dataset_io import combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public", config="default", seed=42)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", config="holdout", seed=99)
    out = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=out, write_prepared_arrow=False)

    md = json.loads((out / "metadata.json").read_text())
    assert md["default_split"] == "public"
    assert md["splits"]["public"]["config"] == "default"
    assert md["splits"]["holdout"]["config"] == "holdout"

    yaml_frontmatter = (out / "README.md").read_text().split("---", 2)[1]
    assert "config_name: default" in yaml_frontmatter
    assert "config_name: holdout" in yaml_frontmatter
    default_idx = yaml_frontmatter.find("config_name: default")
    holdout_idx = yaml_frontmatter.find("config_name: holdout")
    default_block = yaml_frontmatter[default_idx:holdout_idx]
    holdout_block = yaml_frontmatter[holdout_idx:]
    assert "split: public" in default_block
    assert "split: holdout" not in default_block
    assert "split: holdout" in holdout_block


def test_combine_splits_keeps_multiple_default_config_inputs_legacy_shape(tmp_path: Path):
    from whestbench.dataset_io import combine_split_datasets

    mini = _bake_single_split(tmp_path, "mini", split="mini", config="default", seed=42)
    full = _bake_single_split(tmp_path, "full", split="full", config="default", seed=99)
    out = tmp_path / "combined"
    combine_split_datasets([mini, full], output_dir=out, write_prepared_arrow=False)

    md = json.loads((out / "metadata.json").read_text())
    assert "default_split" not in md
    yaml_frontmatter = (out / "README.md").read_text().split("---", 2)[1]
    assert yaml_frontmatter.count("config_name:") == 1
    assert "config_name: default" in yaml_frontmatter
    assert "split: mini" in yaml_frontmatter
    assert "split: full" in yaml_frontmatter


def test_combine_splits_omits_default_split_when_unset(tmp_path: Path):
    """Without an explicit default_split, the field is absent from metadata.json."""
    from whestbench.dataset_io import combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public", seed=42)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=99)
    out = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=out)

    md = json.loads((out / "metadata.json").read_text())
    assert "default_split" not in md


def test_combine_splits_rejects_unknown_default_split(tmp_path: Path):
    """default_split must name one of the input splits."""
    from whestbench.dataset_io import MergeIncompatibleError, combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public", seed=42)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=99)
    with pytest.raises(MergeIncompatibleError, match=r"default_split.*not one of"):
        combine_split_datasets(
            [pub, hold],
            output_dir=tmp_path / "x",
            default_split="nonexistent",
        )


# -----------------------------------------------------------------------------
# prepared_splits (Arrow fast path)
# -----------------------------------------------------------------------------


def test_combine_splits_writes_prepared_arrow_by_default(tmp_path: Path):
    """combine_split_datasets emits prepared/<split>/ for each split by default."""
    from whestbench.dataset_io import combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public", seed=42)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=99)
    out = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=out)

    # Filesystem layout per `Dataset.save_to_disk()` contract.
    for split in ("public", "holdout"):
        prep_dir = out / "prepared" / split
        assert prep_dir.is_dir(), f"missing prepared dir for {split}"
        assert (prep_dir / "dataset_info.json").is_file()
        assert (prep_dir / "state.json").is_file()
        arrow_shards = list(prep_dir.glob("data-*.arrow"))
        assert len(arrow_shards) >= 1, f"no .arrow shards under {prep_dir}"

    # Metadata block.
    md = json.loads((out / "metadata.json").read_text())
    assert "prepared_splits" in md
    assert md["prepared_splits"] == {
        "public": {"path": "prepared/public", "format": "save_to_disk"},
        "holdout": {"path": "prepared/holdout", "format": "save_to_disk"},
    }


def test_combine_splits_skips_prepared_arrow_when_disabled(tmp_path: Path):
    """write_prepared_arrow=False omits the prepared tree and the metadata block."""
    from whestbench.dataset_io import combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public", seed=42)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=99)
    out = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=out, write_prepared_arrow=False)

    assert not (out / "prepared").exists()
    md = json.loads((out / "metadata.json").read_text())
    assert "prepared_splits" not in md


def test_prepared_arrow_is_loadable_via_load_from_disk(tmp_path: Path):
    """The prepared dir round-trips through datasets.load_from_disk."""
    from datasets import Dataset, load_from_disk

    from whestbench.dataset_io import combine_split_datasets

    pub = _bake_single_split(tmp_path, "pub", split="public", seed=42)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=99)
    out = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=out)

    for split, expected_n in (("public", 2), ("holdout", 2)):
        ds = load_from_disk(str(out / "prepared" / split))
        assert isinstance(ds, Dataset)
        assert len(ds) == expected_n


def test_write_prepared_arrow_split_rejects_existing_output(tmp_path: Path):
    from whestbench.dataset_io import combine_split_datasets, write_prepared_arrow_split

    pub = _bake_single_split(tmp_path, "pub", split="public", seed=42)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=99)
    out = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=out)

    parquet = out / "data" / "public-00000-of-00001.parquet"
    existing = out / "prepared" / "public"
    assert existing.exists()
    with pytest.raises(FileExistsError):
        write_prepared_arrow_split(parquet, existing, split="public")


def test_write_prepared_arrow_split_handles_multi_shard(tmp_path: Path):
    """Multi-shard parquet (e.g. full-00000-of-00008.parquet ...) round-trips."""
    from datasets import Dataset, load_from_disk

    from whestbench.dataset_io import combine_split_datasets, write_prepared_arrow_split

    pub = _bake_single_split(tmp_path, "pub", split="public", seed=42)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=99)
    out = tmp_path / "combined"
    # Build via combine first (gets us a real parquet to split).
    combine_split_datasets([pub, hold], output_dir=out, write_prepared_arrow=False)

    # Fake a multi-shard layout by splitting one parquet into two.
    src_parquet = out / "data" / "public-00000-of-00001.parquet"
    ds = Dataset.from_parquet(str(src_parquet))
    half = len(ds) // 2 or 1
    shard0 = ds.select(range(0, half))
    shard1 = ds.select(range(half, len(ds)))
    shard0_path = out / "data" / "public-00000-of-00002.parquet"
    shard1_path = out / "data" / "public-00001-of-00002.parquet"
    shard0.to_parquet(str(shard0_path))
    shard1.to_parquet(str(shard1_path))

    target = tmp_path / "prepared_multi"
    write_prepared_arrow_split([shard1_path, shard0_path], target, split="public")
    materialised = load_from_disk(str(target))
    assert isinstance(materialised, Dataset)
    # The result row count must equal the union of the two shards.
    assert len(materialised) == len(ds)


def test_build_prepared_splits_for_directory_mutates_metadata(tmp_path: Path):
    """build_prepared_splits_for_directory writes the metadata block in-place."""
    from whestbench.dataset_io import (
        build_prepared_splits_for_directory,
        combine_split_datasets,
    )

    pub = _bake_single_split(tmp_path, "pub", split="public", seed=42)
    hold = _bake_single_split(tmp_path, "hold", split="holdout", seed=99)
    out = tmp_path / "combined"
    # First build WITHOUT prepared arrow, then patch it on.
    combine_split_datasets([pub, hold], output_dir=out, write_prepared_arrow=False)
    md = json.loads((out / "metadata.json").read_text())
    assert "prepared_splits" not in md

    result = build_prepared_splits_for_directory(out, splits=["public", "holdout"], metadata=md)
    # Returned mapping equals the mutated block.
    assert result == md["prepared_splits"]
    assert result == {
        "public": {"path": "prepared/public", "format": "save_to_disk"},
        "holdout": {"path": "prepared/holdout", "format": "save_to_disk"},
    }
    # Files exist on disk.
    assert (out / "prepared" / "public" / "state.json").is_file()
    assert (out / "prepared" / "holdout" / "state.json").is_file()
