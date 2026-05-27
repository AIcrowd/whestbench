"""Verify push/pull/inspect aliases still work and emit deprecation warnings.

The Phase 6 rename moved `push → upload`, `pull → download`, `inspect → info`.
The old names continue to function as argparse aliases for one minor release;
each invocation must:

1. Print a one-line deprecation warning that names both the deprecated and the
   canonical verb.
2. Dispatch to the canonical handler (i.e. actually run the requested op).

These tests stub out the heavyweight underlying ops (HF upload, HF
snapshot_download, HF preflight, on-disk metadata reads) so the harness only
exercises the alias path itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest
from rich.console import Console as _RichConsole

import whestbench.cli as cli
import whestbench.dataset_io as _dataset_io_mod
import whestbench.hf_progress as _hf_progress_mod
import whestbench.hub as _hub_mod
from whestbench.hf_progress import HFPreflight


def _spy_console_print(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Capture every ``Console.print`` first-arg string; return the list.

    Real print still fires (so test failures show the captured copy in pytest's
    captured stdout/stderr).
    """
    captured: List[str] = []
    original_print = _RichConsole.print

    def spy_print(self: _RichConsole, *args: Any, **kwargs: Any) -> Any:
        if args:
            captured.append(str(args[0]))
        return original_print(self, *args, **kwargs)

    monkeypatch.setattr(_RichConsole, "print", spy_print)
    return captured


def test_alias_push_dispatches_to_upload_and_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`whest dataset push` is alias for `upload`; emits deprecation + runs upload."""
    captured = _spy_console_print(monkeypatch)

    publish_calls: List[dict] = []

    def fake_publish(local_dir: Any, **kwargs: Any) -> str:
        publish_calls.append({"local_dir": str(local_dir), **kwargs})
        return "deadbeef" * 5

    monkeypatch.setattr(_hub_mod, "publish_dataset", fake_publish)

    # The hf_upload context manager monkey-patches huggingface_hub's tqdm; it's
    # harmless to let it run with our stubbed publish_dataset (no upload work
    # actually happens). Still, build a real-shaped dir so the preflight summary
    # has something to measure.
    local = tmp_path / "ds"
    (local / "data").mkdir(parents=True)
    (local / "metadata.json").write_text("{}")
    (local / "data" / "public-00000-of-00001.parquet").write_bytes(b"\0" * 100)

    rc = cli.main(
        [
            "dataset",
            "push",
            str(local),
            "--repo",
            "aicrowd/test",
            "--tag",
            "v1",
        ]
    )

    assert rc == 0
    joined = "\n".join(captured)
    assert "deprecated" in joined.lower(), f"missing deprecation warning; got: {joined!r}"
    assert "whest dataset upload" in joined, f"warning should point to canonical; got: {joined!r}"
    # The intent line for upload should still fire (alias dispatches into the
    # upload branch).
    assert "Uploading" in joined, f"upload intent missing; got: {joined!r}"
    assert len(publish_calls) == 1, f"publish_dataset should fire exactly once; got {publish_calls}"
    assert publish_calls[0]["repo_id"] == "aicrowd/test"
    assert publish_calls[0]["tag"] == "v1"


def test_alias_pull_dispatches_to_download_and_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`whest dataset pull` is alias for `download`; emits deprecation + runs download."""
    captured = _spy_console_print(monkeypatch)

    # Stub preflight so we don't hit the network. Cache-hit path keeps
    # `hf_download` cheap (status spinner only — no tqdm patch).
    fake_preflight = HFPreflight(
        repo_id="aicrowd/test",
        revision="v1",
        file_count=2,
        total_bytes=2048,
        is_cached=True,
        files=[("metadata.json", 48), ("data/public-00000-of-00001.parquet", 2000)],
    )
    monkeypatch.setattr(_hf_progress_mod, "hf_preflight", lambda *_a, **_k: fake_preflight)

    snapshot_calls: List[dict] = []

    out_dir = tmp_path / "pulled"

    def fake_snapshot_download(**kwargs: Any) -> str:
        snapshot_calls.append(kwargs)
        # Pretend the download placed the dir; create it so the on-disk
        # size measurement in the download branch doesn't fail.
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    # snapshot_download is imported INSIDE the download dispatch branch via
    # `from huggingface_hub import snapshot_download`. Patch on the source
    # module so the in-branch import resolves to our fake.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    rc = cli.main(
        [
            "dataset",
            "pull",
            "aicrowd/test",
            "--revision",
            "v1",
            "--output",
            str(out_dir),
        ]
    )

    assert rc == 0
    joined = "\n".join(captured)
    assert "deprecated" in joined.lower(), f"missing deprecation warning; got: {joined!r}"
    assert "whest dataset download" in joined, f"warning should point to canonical; got: {joined!r}"
    assert "Downloading hf://aicrowd/test@v1" in joined, f"download intent missing; got: {joined!r}"
    assert len(snapshot_calls) == 1, (
        f"snapshot_download should fire exactly once; got {snapshot_calls}"
    )


def test_alias_inspect_dispatches_to_info_and_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`whest dataset inspect` is alias for `info`; emits deprecation + runs info."""
    captured = _spy_console_print(monkeypatch)

    # Build a minimal local dataset dir + metadata.json so read_metadata works.
    local = tmp_path / "ds"
    local.mkdir()
    metadata_calls: List[Any] = []

    def fake_read_metadata(p: Any) -> dict:
        metadata_calls.append(p)
        return {
            "schema_version": "3.0",
            "format": "parquet",
            "backend": "flopscope",
            "n_mlps": 4,
            "n_samples": 100,
            "width": 4,
            "depth": 2,
            "created_at_utc": "2026-05-27T00:00:00+00:00",
            "seed_protocol": {
                "name": "whestbench_explicit_per_mlp_seeds",
                "version": "3.0",
            },
        }

    monkeypatch.setattr(_dataset_io_mod, "read_metadata", fake_read_metadata)

    rc = cli.main(["dataset", "inspect", str(local)])

    assert rc == 0
    joined = "\n".join(captured)
    assert "deprecated" in joined.lower(), f"missing deprecation warning; got: {joined!r}"
    assert "whest dataset info" in joined, f"warning should point to canonical; got: {joined!r}"
    assert len(metadata_calls) == 1, f"read_metadata should fire exactly once; got {metadata_calls}"


@pytest.mark.parametrize(
    "deprecated,canonical",
    [("push", "upload"), ("pull", "download"), ("inspect", "info")],
)
def test_deprecation_warning_mentions_v07_removal(
    deprecated: str, canonical: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deprecation message tells users when aliases will be removed."""
    captured = _spy_console_print(monkeypatch)

    # Build a fake args namespace and call the dispatcher directly. This skips
    # all heavyweight subcommand handlers (the dispatcher returns 2 / prints
    # 'Unknown' if it can't resolve, but our patched dispatch detection runs
    # BEFORE the subcommand handlers, so we can short-circuit by patching the
    # canonical branch to raise SystemExit(0) after the warning fires).
    from argparse import Namespace

    # The canonical handlers all read various args. Trip them to short-circuit
    # via SystemExit so we never have to populate full arg shapes.
    def _bail(*_a: Any, **_k: Any) -> Any:
        raise SystemExit(0)

    # Patch every downstream entrypoint the canonical branches reach for.
    # `info` branches on `Path(src).exists()` — if False it hits hf_hub_download,
    # so bail there too (otherwise validate_repo_id rejects our path-shaped src).
    monkeypatch.setattr(_hub_mod, "publish_dataset", _bail)
    monkeypatch.setattr(_dataset_io_mod, "read_metadata", _bail)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _bail)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _bail)
    monkeypatch.setattr(_hf_progress_mod, "hf_preflight", lambda *_a, **_k: None)

    ns = Namespace(
        dataset_cmd=deprecated,
        # populate every attr the canonical branches *might* read before the
        # bail point so getattr() lookups don't AttributeError.
        local_dir="/tmp/dummy",
        repo="aicrowd/x",
        tag=None,
        private=False,
        token=None,
        message=None,
        repo_id="aicrowd/x",
        revision=None,
        output="/tmp/out",
        split=None,
        source="/tmp/dummy",
    )

    try:
        cli._dispatch_dataset_command(ns)
    except SystemExit:
        pass

    joined = "\n".join(captured)
    assert "deprecated" in joined.lower(), (
        f"alias {deprecated!r} should emit deprecation; got: {joined!r}"
    )
    assert canonical in joined, (
        f"deprecation should name canonical verb {canonical!r}; got: {joined!r}"
    )
    assert "v0.7" in joined, f"deprecation should name removal version; got: {joined!r}"
