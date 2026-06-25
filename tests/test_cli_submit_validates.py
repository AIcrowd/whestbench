"""`whest submit <bad-tarball>` aborts on local validation before any upload."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import whestbench.aicrowd_config as cfg
import whestbench.cli as cli


def _bad_tarball(path: Path) -> None:
    # manifest lists a directory entry — the #107 crash shape
    manifest = {
        "schema_version": "1.0",
        "api_version": "2.0",
        "entrypoint": {"module": "estimator", "class": "Estimator"},
        "files": [{"name": "arc_tools/", "sha256": "b8c5ca68"}],
    }
    body = b"x = 1\n"
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo("estimator.py")
        info.size = len(body)
        tf.addfile(info, io.BytesIO(body))
        blob = json.dumps(manifest).encode("utf-8")
        mi = tarfile.TarInfo("manifest.json")
        mi.size = len(blob)
        tf.addfile(mi, io.BytesIO(blob))


def test_submit_aborts_on_invalid_archive_before_upload(tmp_path: Path, monkeypatch) -> None:
    bad = tmp_path / "bad.tar.gz"
    _bad_tarball(bad)

    # Make auth succeed locally and ensure the network client is never constructed.
    monkeypatch.setattr(cfg, "resolve_api_key", lambda *a, **k: "dummy-key")

    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("AIcrowdClient must not be constructed for an invalid archive")

    monkeypatch.setattr(cli, "AIcrowdClient", _boom)

    rc = cli.main(["submit", str(bad), "--json"])
    assert rc == 2
