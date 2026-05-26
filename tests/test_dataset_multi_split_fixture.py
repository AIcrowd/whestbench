"""Regression test for the frozen multi-split v3.0 fixture.

Catches accidental on-disk format drift in future PRs.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "multi_split_v3.0"


def test_fixture_directory_layout():
    assert (FIXTURE / "data" / "public-00000-of-00001.parquet").is_file()
    assert (FIXTURE / "data" / "holdout-00000-of-00001.parquet").is_file()
    assert (FIXTURE / "metadata.json").is_file()
    assert (FIXTURE / "README.md").is_file()


def test_fixture_metadata_shape():
    md = json.loads((FIXTURE / "metadata.json").read_text())
    assert md["schema_version"] == "3.0"
    assert md["format"] == "hf-datasets-parquet"
    assert md["backend"] == "flopscope"
    assert md["width"] == 4
    assert md["depth"] == 2
    assert md["n_samples"] == 100
    assert set(md["splits"].keys()) == {"public", "holdout"}
    assert md["splits"]["public"]["n_mlps"] == 4
    assert md["splits"]["public"]["seed"] == 1
    assert md["splits"]["holdout"]["n_mlps"] == 4
    assert md["splits"]["holdout"]["seed"] == 2


def test_fixture_loads_via_whestbench():
    from datasets import DatasetDict

    from whestbench.dataset import load_dataset
    from whestbench.dataset import metadata as md_accessor

    dsd = load_dataset(FIXTURE)
    assert isinstance(dsd, DatasetDict)
    assert len(dsd["public"]) == 4
    assert len(dsd["holdout"]) == 4

    md = md_accessor(dsd)
    assert "splits" in md
    assert sorted(md["splits"]) == ["holdout", "public"]

    # Row-level content checks — catch serialization bugs that preserve shape but shuffle values.
    assert dsd["public"]["mlp_id"] == [0, 1, 2, 3]
    assert dsd["holdout"]["mlp_id"] == [0, 1, 2, 3]
    # Every MLP has a non-empty, hyphenated name (faker slug).
    for split_ds in (dsd["public"], dsd["holdout"]):
        for name in split_ds["mlp_name"]:
            assert isinstance(name, str) and "-" in name and name


def test_fixture_member_dataset_supports_iter_mlps():
    from whestbench.dataset import iter_mlps, load_dataset

    dsd = load_dataset(FIXTURE)
    mlps_pub = list(iter_mlps(dsd["public"]))
    assert len(mlps_pub) == 4
    for m in mlps_pub:
        m.validate()
        assert m.width == 4
        assert m.depth == 2
