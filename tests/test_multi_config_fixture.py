"""Regression test for the frozen two-config (config-per-split) fixture.

Catches accidental on-disk format/layout drift. Regenerate with:
    uv run python tests/fixtures/_build_multi_config_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "multi_config_v3.0"


def test_fixture_directory_layout():
    assert (FIXTURE / "data" / "public-00000-of-00001.parquet").is_file()
    assert (FIXTURE / "data" / "holdout-00000-of-00001.parquet").is_file()
    assert (FIXTURE / "metadata.json").is_file()
    assert (FIXTURE / "README.md").is_file()
    assert not (FIXTURE / "prepared").exists()


def test_fixture_metadata_carries_per_split_configs():
    md = json.loads((FIXTURE / "metadata.json").read_text())
    assert md["schema_version"] == "3.0"
    assert md["width"] == 8
    assert md["depth"] == 2
    assert set(md["splits"].keys()) == {"public", "holdout"}
    assert md["splits"]["public"]["config"] == "default"
    assert md["splits"]["holdout"]["config"] == "holdout"
    assert md["default_split"] == "public"


def test_fixture_readme_emits_two_configs():
    frontmatter = (FIXTURE / "README.md").read_text().split("---", 2)[1]
    assert "config_name: default" in frontmatter
    assert "config_name: holdout" in frontmatter


def test_fixture_loads_via_whestbench():
    from datasets import DatasetDict

    from whestbench.dataset import load_dataset

    dsd = load_dataset(FIXTURE)
    assert isinstance(dsd, DatasetDict)
    assert len(dsd["public"]) == 2
    assert len(dsd["holdout"]) == 2
