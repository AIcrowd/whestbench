"""Unit tests for whestbench.hf_progress."""

from __future__ import annotations

from dataclasses import is_dataclass

from whestbench.hf_progress import HFPreflight


def test_hfpreflight_is_frozen_dataclass() -> None:
    p = HFPreflight(
        repo_id="aicrowd/foo",
        revision="v1-warmup",
        file_count=3,
        total_bytes=2_000_000_000,
        is_cached=False,
        files=[
            ("data/public-00000-of-00001.parquet", 1_998_000_000),
            ("metadata.json", 6210),
            ("README.md", 13818),
        ],
    )
    assert is_dataclass(p)
    assert p.repo_id == "aicrowd/foo"
    assert p.file_count == 3
    assert p.total_bytes == 2_000_000_000
    assert p.is_cached is False
    assert len(p.files) == 3
