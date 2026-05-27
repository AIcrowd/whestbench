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
    load_dataset_return: Any = None,
) -> List[str]:
    """Patch everything in cli.py's hf:// run path; capture Rich prints.

    Returns the ``captured`` list — every ``Console.print`` call's first
    positional argument is appended as a string before the original print
    runs (so we still see real output if the test fails).

    ``load_dataset_return``: if not ``None``, the mocked ``load_dataset``
    returns this object instead of the default in-memory ``_FakeDS`` stub.
    Useful for handing back a real ``IterableDataset`` for streaming-path
    coverage. The caller is responsible for attaching dataset metadata via
    ``_dataset_mod._METADATA_BY_DS`` if needed.
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

    fake_ds: Any = load_dataset_return if load_dataset_return is not None else _FakeDS()

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


def _assert_say_ordering(
    joined: str,
    *,
    mode: str,
) -> None:
    """Assert ``say.step`` → ``say.intent`` → ``say.ok`` order in captured output.

    For the cache-hit path there is no ``say.intent`` (the ``Loading from
    cache`` banner is a transient ``console.status`` and doesn't land in
    captured prints), so only the present markers are ordered.
    """
    positions = {
        "step": joined.find("Resolving"),
        "intent": joined.find("Downloading") if mode != "cache_hit" else -1,
        "ok": joined.find("✓"),
    }
    present = [(name, pos) for name, pos in positions.items() if pos != -1]
    sorted_present = sorted(present, key=lambda x: x[1])
    assert [n for n, _ in present] == [n for n, _ in sorted_present], (
        f"expected step → intent → ok order; got {present!r} in joined={joined!r}"
    )


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
    _assert_say_ordering(joined, mode="cache_hit")


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
    _assert_say_ordering(joined, mode="cache_miss")


def test_cli_run_hf_preflight_unavailable_emits_intent_without_size(
    monkeypatch: pytest.MonkeyPatch, _estimator_file: Path
) -> None:
    """When preflight fails (returns ``None``), the download intent should
    still fire — just without a file count / byte total — and the cache-miss
    ✓ line should fire without a byte label.
    """
    captured = _install_run_mocks(monkeypatch, preflight=None)

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
    assert "preflight unavailable" in joined, (
        f"missing preflight-unavailable annotation; got: {joined!r}"
    )
    # ✓ line fires; with no preflight there is no byte count between
    # "Downloaded" and "and loaded".
    assert "✓" in joined and "Downloaded" in joined and "loaded" in joined, (
        f"missing cache-miss ✓ line; got: {joined!r}"
    )
    assert "Downloaded and loaded" in joined, (
        f"preflight-None ok line should read 'Downloaded and loaded ...'; got: {joined!r}"
    )
    _assert_say_ordering(joined, mode="cache_miss")


def test_cli_run_hf_streaming_emits_warning(
    monkeypatch: pytest.MonkeyPatch, _estimator_file: Path
) -> None:
    """--streaming opts into IterableDataset mode and emits the cache trade-off warning."""
    # Mocked preflight is cold cache — streaming should override it.
    fake_preflight = HFPreflight(
        repo_id="aicrowd/foo",
        revision="v1-warmup",
        file_count=1,
        total_bytes=95_000_000,
        is_cached=False,
        files=[("data/public-00000-of-00001.parquet", 95_000_000)],
    )

    # Build a real IterableDataset via Dataset.to_iterable_dataset so cli.py's
    # `isinstance(ds, IterableDataset)` check fires correctly. The single
    # placeholder row is never iterated (we mock make_contest_from_dataset).
    from datasets import Dataset as _RealDataset

    real_iter = _RealDataset.from_list([{"placeholder": 0}]).to_iterable_dataset()
    # Attach dataset metadata via the public weakref side-channel so cli.py's
    # `_wb_metadata(ds)` and `ds_n_mlps` resolution both succeed.
    _dataset_mod._METADATA_BY_DS[real_iter] = {
        "schema_version": "3.0",
        "format": "parquet",
        "backend": "flopscope",
        "n_mlps": 5,
        "n_samples": 10,
        "width": 8,
        "depth": 2,
        "seed_protocol": {
            "name": "whestbench_explicit_per_mlp_seeds",
            "version": "3.0",
        },
    }

    captured = _install_run_mocks(
        monkeypatch,
        preflight=fake_preflight,
        load_dataset_return=real_iter,
    )

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            str(_estimator_file),
            "--dataset",
            "hf://aicrowd/foo@v1-warmup",
            "--n-mlps",
            "1",
            "--streaming",
            "--format",
            "plain",
        ]
    )

    assert exit_code == 0, f"streaming run unexpectedly failed; got: {exit_code!r}"
    joined = "\n".join(captured)
    assert "Streaming from HF" in joined, f"missing streaming banner; got: {joined!r}"
    assert "Iteration-only" in joined, f"missing iteration-only line in warning; got: {joined!r}"
    assert "Not cached locally" in joined, (
        f"missing cache-trade-off line in warning; got: {joined!r}"
    )
    # The forward-pointing hint must use the new verb name *and* include the
    # exact repo + revision the user asked for, so they can copy-paste it.
    assert "whest dataset download aicrowd/foo --revision v1-warmup" in joined, (
        f"missing copy-pasteable populate-cache hint; got: {joined!r}"
    )
    # Streaming-specific ok line.
    assert "Streaming dataset ready" in joined, f"missing streaming ok line; got: {joined!r}"
    # Sanity: the cache-miss intent line must NOT fire (streaming overrides it).
    assert "Downloading hf://aicrowd/foo@v1-warmup" not in joined, (
        f"unexpected download intent in streaming mode; got: {joined!r}"
    )


def test_cli_run_streaming_with_local_path_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--streaming on a local-path dataset fails fast with a clear message."""
    # Any existing path qualifies as a local dataset arg for _resolve_dataset_arg.
    local = tmp_path / "fake-eval"
    local.mkdir()

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            "some-est.py",  # never reached: we bail before estimator loading.
            "--dataset",
            str(local),
            "--streaming",
        ]
    )

    assert exit_code != 0, "expected non-zero exit when --streaming is used locally"
    captured = capsys.readouterr()
    combined = (captured.err + captured.out).lower()
    assert "streaming" in combined, (
        f"missing 'streaming' in error output; got err={captured.err!r} out={captured.out!r}"
    )
    assert "hf://" in combined, (
        f"missing 'hf://' guidance in error output; got err={captured.err!r} out={captured.out!r}"
    )
