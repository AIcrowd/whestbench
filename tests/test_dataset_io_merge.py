"""Tests for dataset_io.merge_datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _bake_partial(
    tmp_path: Path,
    name: str,
    *,
    mlp_range: "tuple[int, int]",
    n_mlps: int = 8,
    mlp_seeds: "list[int] | None" = None,
):
    from whestbench.dataset import create_dataset

    if mlp_seeds is None:
        mlp_seeds = [42 * 1000 + i for i in range(n_mlps)]
    out = tmp_path / name
    create_dataset(
        n_mlps=n_mlps,
        n_samples=50,
        width=4,
        depth=2,
        mlp_seeds=mlp_seeds,
        output_path=out,
        mlp_range=mlp_range,
    )
    return out


def test_merge_two_partials_produces_single_dataset(tmp_path: Path):
    from datasets import load_dataset as hf_load

    from whestbench.dataset_io import merge_datasets

    p0 = _bake_partial(tmp_path, "p0", mlp_range=(0, 4))
    p1 = _bake_partial(tmp_path, "p1", mlp_range=(4, 8))
    merged = tmp_path / "merged"
    merge_datasets([p0, p1], output_dir=merged)
    ds = hf_load(str(merged), split="public")
    assert len(ds) == 8
    assert ds["mlp_id"] == list(range(8))


def test_merge_bit_equivalent_to_single_bake(tmp_path: Path):
    """Merge of 3 partials matches a single-host bake of the same size."""
    from datasets import load_dataset as hf_load

    from whestbench.dataset import create_dataset
    from whestbench.dataset_io import merge_datasets

    mlp_seeds = [7 * 1000 + i for i in range(9)]
    single_out = tmp_path / "single"
    create_dataset(
        n_mlps=9, n_samples=50, width=4, depth=2, mlp_seeds=mlp_seeds, output_path=single_out
    )

    partials = []
    for k in range(3):
        out = _bake_partial(
            tmp_path,
            f"p{k}",
            mlp_range=(3 * k, 3 * (k + 1)),
            n_mlps=9,
            mlp_seeds=mlp_seeds,
        )
        partials.append(out)
    merged = tmp_path / "merged"
    merge_datasets(partials, output_dir=merged)

    s = hf_load(str(single_out), split="public")
    m = hf_load(str(merged), split="public")

    np.testing.assert_array_equal(np.array(s["weights"]), np.array(m["weights"]))
    np.testing.assert_array_equal(np.array(s["all_layer_means"]), np.array(m["all_layer_means"]))
    assert s["mlp_seed"] == m["mlp_seed"]
    assert s["mlp_name"] == m["mlp_name"]
    assert s["mlp_id"] == m["mlp_id"]


def test_merge_rejects_overlapping_ranges(tmp_path: Path):
    from whestbench.dataset_io import MergeOverlapError, merge_datasets

    p0 = _bake_partial(tmp_path, "p0", mlp_range=(0, 5))
    p1 = _bake_partial(tmp_path, "p1", mlp_range=(3, 8))
    with pytest.raises(MergeOverlapError):
        merge_datasets([p0, p1], output_dir=tmp_path / "out")


def test_merge_rejects_gap(tmp_path: Path):
    from whestbench.dataset_io import MergeIncompleteError, merge_datasets

    p0 = _bake_partial(tmp_path, "p0", mlp_range=(0, 3))
    p1 = _bake_partial(tmp_path, "p1", mlp_range=(5, 8), n_mlps=8)
    with pytest.raises(MergeIncompleteError):
        merge_datasets([p0, p1], output_dir=tmp_path / "out")


def test_merge_rejects_different_backends(tmp_path: Path):
    """Merge must reject partials with mismatched bake parameters.

    Note: under seed_protocol 3.0, per-partial seeds are not in metadata
    (they live in the parquet mlp_seed column). `merge_datasets` cannot
    detect "different seeds" at merge time. Mismatched seeds would
    produce a dataset with discontinuous mlp_id values — which IS
    detected by merge_datasets's separate gap/overlap check, not by a
    direct seed comparison. This test covers the explicit-metadata-field
    rejection path (backend); the gap-detection path is tested
    elsewhere.
    """
    import json

    from whestbench.dataset_io import MergeIncompatibleError, merge_datasets

    p0 = _bake_partial(tmp_path, "p0", mlp_range=(0, 4))
    p1 = _bake_partial(tmp_path, "p1", mlp_range=(4, 8))
    # Patch p1 to report a different backend.
    md_path = p1 / "metadata.json"
    md = json.loads(md_path.read_text())
    md["backend"] = "torch"
    md_path.write_text(json.dumps(md, indent=2))
    with pytest.raises(MergeIncompatibleError):
        merge_datasets([p0, p1], output_dir=tmp_path / "out")


def test_merge_rejects_complete_dataset(tmp_path: Path):
    """A complete (non-partial) dataset cannot be merged."""
    from whestbench.dataset import create_dataset
    from whestbench.dataset_io import MergeIncompatibleError, merge_datasets

    complete = tmp_path / "complete"
    mlp_seeds = [1000 + i for i in range(4)]
    create_dataset(
        n_mlps=4, n_samples=50, width=4, depth=2, mlp_seeds=mlp_seeds, output_path=complete
    )
    p1 = _bake_partial(tmp_path, "p1", mlp_range=(0, 4), n_mlps=4, mlp_seeds=mlp_seeds)
    with pytest.raises(MergeIncompatibleError):
        merge_datasets([complete, p1], output_dir=tmp_path / "out")


def test_merged_metadata_has_no_is_partial(tmp_path: Path):
    from whestbench.dataset_io import merge_datasets

    p0 = _bake_partial(tmp_path, "p0", mlp_range=(0, 4))
    p1 = _bake_partial(tmp_path, "p1", mlp_range=(4, 8))
    merged = tmp_path / "merged"
    merge_datasets([p0, p1], output_dir=merged)
    md = json.loads((merged / "metadata.json").read_text())
    assert "is_partial" not in md
    assert "mlp_range" not in md
    assert "total_n_mlps" not in md
    assert "merged_at_utc" in md
    assert "hardware_fingerprints" in md
    assert len(md["hardware_fingerprints"]) == 2
