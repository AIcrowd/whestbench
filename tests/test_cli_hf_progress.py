"""End-to-end-ish tests for the hf:// dataset progress wrapper in cli.py.

These tests patch the heavyweight `_wb_load_dataset` call so we never touch
the network. They assert that the Rich console writes the expected before/
during/after copy in each mode (cache hit, cache miss).

Notes on test design:

* We do NOT pass ``--json``: ``json_output=True`` silences every ``say.*``
  call via ``quiet=json_output`` in cli.py, which would defeat the whole
  point of these tests. Instead we use ``--format plain`` so the Rich Live
  dashboard is skipped (keeping the test cheap) while ``say.*`` still writes
  to the Rich ``_console`` we patch.

* We monkeypatch ``rich.console.Console.print`` globally and collect every
  rendered line into ``captured``. That captures both our wrapper's
  ``say.step`` / ``say.intent`` / ``say.ok`` messages and any incidental
  prints from elsewhere in the run pipeline. We only assert that *specific*
  expected substrings show up.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
from rich.console import Console as _RichConsole

import whestbench.cli as cli
import whestbench.dataset as _dataset_mod
import whestbench.hf_progress as _hf_progress_mod
import whestbench.scoring as _scoring_mod
from whestbench.hf_progress import HFPreflight


def _install_run_mocks(
    monkeypatch: pytest.MonkeyPatch,
    preflight: "HFPreflight | None",
    *,
    n_mlps: int = 3,
) -> List[str]:
    """Patch everything in cli.py's hf:// run path; capture Rich prints.

    Returns the ``captured`` list — every ``Console.print`` call's first
    positional argument is appended as a string before the original print
    runs (so we still see real output if the test fails).
    """
    captured: List[str] = []

    original_print = _RichConsole.print

    def spy_print(self: _RichConsole, *args: Any, **kwargs: Any) -> Any:
        if args:
            captured.append(str(args[0]))
        return original_print(self, *args, **kwargs)

    monkeypatch.setattr(_RichConsole, "print", spy_print)

    # Fake dataset — needs to support ``len()`` and NOT be a DatasetDict.
    # SimpleNamespace can't expose ``__len__`` (immutable type), so use a
    # tiny ad-hoc class.
    class _FakeDS:
        def __len__(self) -> int:
            return n_mlps

    fake_ds = _FakeDS()

    def fake_load_dataset(*_args: Any, **_kwargs: Any) -> Any:
        return fake_ds

    monkeypatch.setattr(_dataset_mod, "load_dataset", fake_load_dataset)

    # Preflight returns the scenario's HFPreflight (or None).
    def fake_preflight(*_args: Any, **_kwargs: Any) -> "HFPreflight | None":
        return preflight

    monkeypatch.setattr(_hf_progress_mod, "hf_preflight", fake_preflight)

    # Attached-metadata lookup; bypass the WeakKeyDict in dataset.metadata.
    monkeypatch.setattr(
        cli,
        "_wb_metadata",
        lambda _ds: {
            "schema_version": "3.0",
            "n_mlps": n_mlps,
            "n_samples": 100,
            "width": 4,
            "depth": 2,
            "seed": 42,
        },
    )
    # sha256 of the metadata file — short-circuit so we don't hit the network.
    monkeypatch.setattr(cli, "_metadata_file_hash", lambda *_a, **_k: "deadbeef")

    # Contest data — only fields the run pipeline reads.
    fake_contest_data = SimpleNamespace(
        spec=None,
        mlps=[],
        all_layer_targets=[],
        final_targets=[],
        avg_variances=[],
        sampling_budget_breakdown=None,
    )
    monkeypatch.setattr(
        _scoring_mod,
        "make_contest_from_dataset",
        lambda *_a, **_k: fake_contest_data,
    )

    # Estimator class metadata — used by the non-json run path.
    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: SimpleNamespace(class_name="Estimator"),
        raising=False,
    )

    # Skip the actual estimator run; return a minimal valid report.
    fake_report = {
        "schema_version": "1.1",
        "mode": "human",
        "detail": "raw",
        "run_meta": {
            "run_started_at_utc": "2026-05-27T00:00:00+00:00",
            "run_finished_at_utc": "2026-05-27T00:00:01+00:00",
            "run_duration_s": 1.0,
        },
        "run_config": {
            "n_mlps": n_mlps,
            "width": 4,
            "depth": 2,
            "flop_budget": 1000,
            "profile_enabled": False,
        },
        "results": {},
        "notes": [],
    }
    monkeypatch.setattr(
        cli,
        "_run_estimator_with_runner",
        lambda *_a, **_k: dict(fake_report),
    )

    # Short-circuit report rendering so we don't have to populate every field.
    monkeypatch.setattr(
        cli,
        "_render_plain_text_report",
        lambda *_a, **_k: "(report rendering stubbed)",
    )

    return captured


@pytest.fixture()
def _estimator_file(tmp_path: Path) -> Path:
    """A minimal estimator file on disk; cli.py only resolves the path."""
    path = tmp_path / "noop_estimator.py"
    path.write_text(
        "from whestbench.sdk import BaseEstimator\n"
        "import flopscope.numpy as fnp\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        return fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)\n"
    )
    return path


def test_cli_run_hf_cache_hit_emits_cache_hit_status(
    monkeypatch: pytest.MonkeyPatch, _estimator_file: Path
) -> None:
    """Cache-hit preflight should yield a ``(from cache)`` ok line."""
    preflight = HFPreflight(
        repo_id="aicrowd/foo",
        revision="v1-warmup",
        file_count=1,
        total_bytes=6000,
        is_cached=True,
        files=[("metadata.json", 6000)],
    )
    captured = _install_run_mocks(monkeypatch, preflight)

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            str(_estimator_file),
            "--dataset",
            "hf://aicrowd/foo@v1-warmup",
            "--format",
            "plain",
        ]
    )

    assert exit_code == 0
    joined = "\n".join(captured)
    assert "from cache" in joined.lower(), f"missing cache-hit message; got: {joined!r}"


def test_cli_run_hf_cache_miss_emits_download_intent(
    monkeypatch: pytest.MonkeyPatch, _estimator_file: Path
) -> None:
    """Cache-miss preflight should announce the download with bytes and title."""
    # 2 GiB so format_bytes (1024-based) renders exactly "2.0 GB".
    two_gib = 2 * 1024**3
    preflight = HFPreflight(
        repo_id="aicrowd/foo",
        revision="v1-warmup",
        file_count=1,
        total_bytes=two_gib,
        is_cached=False,
        files=[("data/public-00000-of-00001.parquet", two_gib)],
    )
    captured = _install_run_mocks(monkeypatch, preflight)

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            str(_estimator_file),
            "--dataset",
            "hf://aicrowd/foo@v1-warmup",
            "--format",
            "plain",
        ]
    )

    assert exit_code == 0
    joined = "\n".join(captured)
    assert "Downloading hf://aicrowd/foo@v1-warmup" in joined, (
        f"missing download intent; got: {joined!r}"
    )
    assert "2.0 GB" in joined, f"missing byte total; got: {joined!r}"
    # Lock the cache-miss ✓ format: a single space between "Downloaded" and
    # the byte total — no stray comma (regression guard).
    assert "Downloaded 2.0 GB" in joined, (
        f"cache-miss ok line should read 'Downloaded 2.0 GB ...'; got: {joined!r}"
    )
