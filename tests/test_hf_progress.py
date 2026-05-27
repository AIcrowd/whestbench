"""Unit tests for whestbench.hf_progress."""

from __future__ import annotations

import io
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import huggingface_hub.utils
import pytest
from rich.console import Console as RichConsole

from whestbench.hf_progress import (
    _ACTIVE_RICH_PROGRESS,  # noqa: F401 — imported to assert the symbol exists
    HFPreflight,
    RichHFTqdm,
    hf_download,
    hf_preflight,
    hf_upload,
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


def _fake_dataset_info(siblings: list[dict]) -> SimpleNamespace:
    """Build a stub that ``HfApi.dataset_info`` returns.

    Each sibling is a dict with rfilename + size. We expose them as attributes
    via ``SimpleNamespace`` so they behave like the real ``RepoSibling`` objects
    without us having to pin to HF Hub's actual class shape.
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


def _live_hf_tqdm() -> object:
    """Read the current ``huggingface_hub.utils.tqdm`` class via getattr.

    The class is not in ``__all__``, so direct attribute access trips Pyright's
    ``reportPrivateImportUsage``. Going through ``getattr`` keeps the type
    checker quiet without sprinkling per-line ignores across the test file.
    """
    return getattr(huggingface_hub.utils, "tqdm")


def test_hf_download_cache_hit_mode_does_not_install_richhftqdm() -> None:
    buf = io.StringIO()
    con = RichConsole(file=buf, force_terminal=True, color_system=None, width=120)
    original = _live_hf_tqdm()
    pf = HFPreflight(
        repo_id="aicrowd/foo",
        revision="v1-warmup",
        file_count=1,
        total_bytes=100,
        is_cached=True,
        files=[("metadata.json", 100)],
    )
    with hf_download(con, title="hf://aicrowd/foo@v1-warmup", preflight=pf, mode="cache_hit"):
        # In cache_hit mode we DO NOT monkey-patch the tqdm class.
        assert _live_hf_tqdm() is original
    assert _live_hf_tqdm() is original  # restored on exit too


def test_hf_download_materialize_mode_swaps_tqdm() -> None:
    buf = io.StringIO()
    con = RichConsole(file=buf, force_terminal=True, color_system=None, width=120)
    original = _live_hf_tqdm()
    pf = HFPreflight(
        repo_id="aicrowd/foo",
        revision="v1-warmup",
        file_count=1,
        total_bytes=2_000_000_000,
        is_cached=False,
        files=[("data/public-00000-of-00001.parquet", 2_000_000_000)],
    )
    with hf_download(con, title="hf://aicrowd/foo@v1-warmup", preflight=pf, mode="materialize"):
        assert _live_hf_tqdm() is not original
    # Restored on exit.
    assert _live_hf_tqdm() is original


def test_hf_download_restores_tqdm_on_exception() -> None:
    con = RichConsole(file=io.StringIO(), force_terminal=True, color_system=None, width=120)
    original = _live_hf_tqdm()
    pf = HFPreflight(
        repo_id="aicrowd/foo",
        revision=None,
        file_count=1,
        total_bytes=2_000_000_000,
        is_cached=False,
        files=[("data/x.parquet", 2_000_000_000)],
    )
    with pytest.raises(RuntimeError):
        with hf_download(con, title="hf://x", preflight=pf, mode="materialize"):
            raise RuntimeError("boom")
    assert _live_hf_tqdm() is original


def test_hf_upload_swaps_tqdm_and_restores(tmp_path: Path) -> None:
    # Make a small fake folder to upload.
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.parquet").write_bytes(b"\0" * 1024)
    (tmp_path / "metadata.json").write_text("{}")

    con = RichConsole(file=io.StringIO(), force_terminal=True, color_system=None, width=120)
    original = _live_hf_tqdm()
    with hf_upload(con, title="hf://aicrowd/foo@v1", local_dir=tmp_path):
        assert _live_hf_tqdm() is not original
    assert _live_hf_tqdm() is original


def test_hf_upload_quiet_is_passthrough(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text("{}")
    con = RichConsole(file=io.StringIO(), force_terminal=True, color_system=None, width=120)
    original = _live_hf_tqdm()
    with hf_upload(con, title="hf://x", local_dir=tmp_path, quiet=True):
        assert _live_hf_tqdm() is original
    assert _live_hf_tqdm() is original
