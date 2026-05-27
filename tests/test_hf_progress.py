"""Unit tests for whestbench.hf_progress."""

from __future__ import annotations

from dataclasses import is_dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from whestbench.hf_progress import (
    _ACTIVE_RICH_PROGRESS,  # noqa: F401 — imported to assert the symbol exists
    HFPreflight,
    RichHFTqdm,
    hf_preflight,
)


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


def _fake_dataset_info(siblings: list[dict]) -> MagicMock:
    """Build a mock that HfApi.dataset_info returns.

    Each sibling is a dict with rfilename + size. We expose them as attributes
    via SimpleNamespace so they behave like the real ``RepoSibling`` objects.
    """
    return SimpleNamespace(
        sha="198ab8a15ad60cb858feb22c59b3b53bb6ae98ec",
        siblings=[SimpleNamespace(**s) for s in siblings],
    )


def test_hf_preflight_filters_to_split_parquet_plus_meta() -> None:
    info = _fake_dataset_info(
        [
            {"rfilename": "data/public-00000-of-00001.parquet", "size": 2_000_000_000},
            {"rfilename": "data/holdout-00000-of-00001.parquet", "size": 1_000_000},
            {"rfilename": "metadata.json", "size": 6000},
            {"rfilename": "README.md", "size": 13_000},
            {"rfilename": ".gitattributes", "size": 2500},
        ]
    )
    with (
        patch("whestbench.hf_progress.HfApi") as MockApi,
        patch("whestbench.hf_progress.try_to_load_from_cache", return_value=None),
    ):
        MockApi.return_value.dataset_info.return_value = info
        pf = hf_preflight("aicrowd/foo", revision="v1-warmup", split="public")
    assert pf is not None
    assert pf.repo_id == "aicrowd/foo"
    assert pf.revision == "v1-warmup"
    rfiles = [n for n, _ in pf.files]
    assert "data/public-00000-of-00001.parquet" in rfiles
    assert "data/holdout-00000-of-00001.parquet" not in rfiles
    assert "metadata.json" in rfiles
    assert "README.md" in rfiles
    assert ".gitattributes" not in rfiles
    assert pf.total_bytes == 2_000_000_000 + 6_000 + 13_000


def test_hf_preflight_reports_cache_hit_when_all_files_cached() -> None:
    info = _fake_dataset_info(
        [
            {"rfilename": "data/public-00000-of-00001.parquet", "size": 2_000_000_000},
            {"rfilename": "metadata.json", "size": 6000},
            {"rfilename": "README.md", "size": 13_000},
        ]
    )

    def _fake_cache_hit(*, repo_id, filename, repo_type, revision, cache_dir=None):  # noqa: ARG001
        # Every file is reported as a cached path
        return f"/tmp/fake-cache/{repo_id}/{filename}"

    with (
        patch("whestbench.hf_progress.HfApi") as MockApi,
        patch("whestbench.hf_progress.try_to_load_from_cache", side_effect=_fake_cache_hit),
    ):
        MockApi.return_value.dataset_info.return_value = info
        pf = hf_preflight("aicrowd/foo", revision="v1-warmup", split="public")
    assert pf is not None
    assert pf.is_cached is True


def test_hf_preflight_returns_none_on_hf_error() -> None:
    with patch("whestbench.hf_progress.HfApi") as MockApi:
        MockApi.return_value.dataset_info.side_effect = ConnectionError("offline")
        pf = hf_preflight("aicrowd/foo", revision="v1-warmup", split="public")
    assert pf is None


def test_richhftqdm_forwards_updates_to_active_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, int]] = []

    class _FakeProgress:
        def add_task(self, description, *, total):  # noqa: ARG002
            events.append(("add_task", total))
            return 7

        def update(self, task_id, *, completed=None, total=None):  # noqa: ARG002
            if completed is not None:
                events.append(("update_completed", completed))

        def remove_task(self, task_id):  # noqa: ARG002
            events.append(("remove", 0))

    fake = _FakeProgress()
    monkeypatch.setattr("whestbench.hf_progress._ACTIVE_RICH_PROGRESS", fake)

    bar = RichHFTqdm(total=1000, desc="test.bin")
    bar.update(250)
    bar.update(750)
    bar.close()

    # add_task fires on __init__, then update_completed twice, then remove on close.
    kinds = [e[0] for e in events]
    assert kinds == ["add_task", "update_completed", "update_completed", "remove"]


def test_richhftqdm_no_active_progress_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("whestbench.hf_progress._ACTIVE_RICH_PROGRESS", None)
    bar = RichHFTqdm(total=10, desc="x")
    bar.update(5)
    bar.close()
    # No exception.
    assert True
