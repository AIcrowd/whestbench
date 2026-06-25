"""Local mirror of the grader's submission-archive integrity check.

Regression for whestbench#107: a manifest files[] directory entry crashes the
grader (IsADirectoryError). validate_package() catches that — and other manifest /
archive drift — locally, before `whest submit` uploads anything."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

from whestbench.packaging import package_submission
from whestbench.validation import validate_package

_EST = (
    "from whestbench import BaseEstimator\n"
    "class Estimator(BaseEstimator):\n"
    "    def predict(self, mlp, budget):\n"
    "        import flopscope.numpy as fnp\n"
    "        return fnp.zeros((mlp.depth, mlp.width))\n"
)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _manifest(
    files: list[dict], *, module: str = "estimator", schema: str = "1.0", api: str = "2.0"
) -> dict:
    return {
        "schema_version": schema,
        "api_version": api,
        "entrypoint": {"module": module, "class": "Estimator"},
        "files": files,
    }


def _write_tarball(
    path: Path, *, file_bytes: dict, manifest: dict, dir_members: tuple = ()
) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in file_bytes.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for d in dir_members:
            info = tarfile.TarInfo(d)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tf.addfile(info)
        blob = json.dumps(manifest, indent=2).encode("utf-8")
        mi = tarfile.TarInfo("manifest.json")
        mi.size = len(blob)
        tf.addfile(mi, io.BytesIO(blob))


def _codes(result) -> set:
    return {i.code for i in result.issues}


def test_valid_archive_passes(tmp_path: Path) -> None:
    body = _EST.encode("utf-8")
    p = tmp_path / "ok.tar.gz"
    _write_tarball(
        p,
        file_bytes={"estimator.py": body},
        manifest=_manifest([{"name": "estimator.py", "sha256": _sha(body)}]),
    )
    result = validate_package(p)
    assert result.ok is True
    assert result.issues == []


def test_real_whest_archive_passes(tmp_path: Path) -> None:
    (tmp_path / "estimator.py").write_text(_EST, encoding="utf-8")
    out = tmp_path / "submission.tar.gz"
    package_submission(tmp_path / "estimator.py", output_path=out)
    assert validate_package(out).ok is True


def test_directory_entry_trailing_slash_rejected(tmp_path: Path) -> None:
    body = _EST.encode("utf-8")
    p = tmp_path / "dir.tar.gz"
    _write_tarball(
        p,
        file_bytes={"estimator.py": body},
        manifest=_manifest(
            [
                {"name": "estimator.py", "sha256": _sha(body)},
                {"name": "arc_tools/", "sha256": "b8c5ca68"},  # the prod #107 shape
            ]
        ),
    )
    result = validate_package(p)
    assert result.ok is False
    assert "directory_entry" in _codes(result)
    assert any("arc_tools" in i.name for i in result.issues)


def test_directory_member_no_slash_rejected(tmp_path: Path) -> None:
    body = _EST.encode("utf-8")
    p = tmp_path / "dir2.tar.gz"
    _write_tarball(
        p,
        file_bytes={"estimator.py": body},
        manifest=_manifest(
            [
                {"name": "estimator.py", "sha256": _sha(body)},
                {"name": "arc_tools", "sha256": "b8c5ca68"},
            ]
        ),
        dir_members=("arc_tools",),
    )
    result = validate_package(p)
    assert result.ok is False
    assert "directory_entry" in _codes(result)


def test_sha256_mismatch_rejected(tmp_path: Path) -> None:
    body = _EST.encode("utf-8")
    p = tmp_path / "drift.tar.gz"
    _write_tarball(
        p,
        file_bytes={"estimator.py": body},
        manifest=_manifest([{"name": "estimator.py", "sha256": "0" * 64}]),
    )
    result = validate_package(p)
    assert result.ok is False
    assert "sha256_mismatch" in _codes(result)


def test_missing_manifest_rejected(tmp_path: Path) -> None:
    p = tmp_path / "nomani.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        data = _EST.encode("utf-8")
        info = tarfile.TarInfo("estimator.py")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    result = validate_package(p)
    assert result.ok is False
    assert "missing_manifest" in _codes(result)


def test_missing_entrypoint_file_rejected(tmp_path: Path) -> None:
    body = b"x = 1\n"
    p = tmp_path / "noentry.tar.gz"
    _write_tarball(
        p,
        file_bytes={"helper.py": body},
        manifest=_manifest([{"name": "helper.py", "sha256": _sha(body)}]),
    )
    result = validate_package(p)
    assert result.ok is False
    assert "missing_entrypoint_file" in _codes(result)


def test_file_not_in_archive_rejected(tmp_path: Path) -> None:
    body = _EST.encode("utf-8")
    p = tmp_path / "missingfile.tar.gz"
    _write_tarball(
        p,
        file_bytes={"estimator.py": body},
        manifest=_manifest(
            [
                {"name": "estimator.py", "sha256": _sha(body)},
                {"name": "helper.py", "sha256": "0" * 64},
            ]
        ),
    )
    result = validate_package(p)
    assert result.ok is False
    assert "file_not_in_archive" in _codes(result)


def test_unsupported_api_version_rejected(tmp_path: Path) -> None:
    body = _EST.encode("utf-8")
    p = tmp_path / "badapi.tar.gz"
    _write_tarball(
        p,
        file_bytes={"estimator.py": body},
        manifest=_manifest([{"name": "estimator.py", "sha256": _sha(body)}], api="9.9"),
    )
    result = validate_package(p)
    assert result.ok is False
    assert "unsupported_api_version" in _codes(result)


def test_not_a_tarball_rejected(tmp_path: Path) -> None:
    p = tmp_path / "junk.tar.gz"
    p.write_bytes(b"not a gzip tar at all")
    result = validate_package(p)
    assert result.ok is False
    assert "archive_unreadable" in _codes(result)


def test_missing_file_rejected(tmp_path: Path) -> None:
    result = validate_package(tmp_path / "does-not-exist.tar.gz")
    assert result.ok is False
    assert "archive_missing" in _codes(result)
