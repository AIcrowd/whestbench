"""CLI transparency for `whest package` / `whest submit --estimator`.

File mode warns it ships only the one file; folder mode loudly lists every file,
notes credential files excluded for security, and (interactively) confirms.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import List

import pytest
from rich.console import Console as _RichConsole

import whestbench.aicrowd_config as cfg
import whestbench.cli as cli

_EST = (
    "from whestbench import BaseEstimator\n"
    "class Estimator(BaseEstimator):\n"
    "    def predict(self, mlp, budget):\n"
    "        import flopscope.numpy as fnp\n"
    "        return fnp.zeros((mlp.depth, mlp.width))\n"
)


def _spy(monkeypatch) -> List[str]:
    captured: List[str] = []
    original = _RichConsole.print

    def spy(self, *args, **kwargs):
        if args:
            captured.append(str(args[0]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_RichConsole, "print", spy)
    return captured


def _archive_names(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as tf:
        return set(tf.getnames())


class _FakeClient:
    def __init__(self, **_kw):
        pass

    def verify_identity(self):
        return 1

    def resolve_challenge(self, _slug):
        return 1

    def check_registration(self, **_kw):
        return True

    def get_upload_details(self, **_kw):
        return {"url": "https://s3.test/upload", "fields": {"key": "k"}}

    def upload_to_s3(self, **_kw):
        return "k"

    def create_submission(self, **_kw):
        return {"data": {"submission_id": 1, "created_at": "t"}}

    def get_submission_status(self, sid):
        return {"id": sid, "grading_status_cd": "graded", "score": 0.5}


def test_package_file_mode_warns_only_that_file(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "estimator.py").write_text(_EST, encoding="utf-8")
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")
    out = tmp_path.parent / "filemode.tar.gz"
    captured = _spy(monkeypatch)

    rc = cli.main(["package", "--estimator", str(tmp_path / "estimator.py"), "--output", str(out)])

    assert rc == 0
    joined = "\n".join(captured)
    assert "single" in joined.lower()
    assert "estimator.py" in joined
    assert _archive_names(out) == {"estimator.py", "manifest.json"}


def test_package_folder_mode_lists_files_and_excludes_secret(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "estimator.py").write_text(_EST, encoding="utf-8")
    (tmp_path / "helper.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (tmp_path / ".env").write_text("AICROWD_API_KEY=secret\n", encoding="utf-8")
    out = tmp_path.parent / "foldermode.tar.gz"
    captured = _spy(monkeypatch)

    rc = cli.main(["package", "--estimator", str(tmp_path), "--output", str(out)])

    assert rc == 0
    joined = "\n".join(captured)
    assert "helper.py" in joined
    assert "requirements.txt" in joined
    assert ".env" in joined and "security" in joined.lower()
    names = _archive_names(out)
    assert ".env" not in names
    assert {"estimator.py", "helper.py", "requirements.txt", "manifest.json"} <= names


@pytest.mark.parametrize("flag", ["--requirements", "--submission-metadata", "--approach"])
def test_package_deprecated_flags_warn(tmp_path: Path, monkeypatch, flag: str) -> None:
    (tmp_path / "estimator.py").write_text(_EST, encoding="utf-8")
    extra = tmp_path / "extra.txt"
    extra.write_text("x\n", encoding="utf-8")
    out = tmp_path.parent / "dep.tar.gz"
    captured = _spy(monkeypatch)

    rc = cli.main(
        [
            "package",
            "--estimator",
            str(tmp_path / "estimator.py"),
            flag,
            str(extra),
            "--output",
            str(out),
        ]
    )

    assert rc == 0
    joined = "\n".join(captured)
    assert flag in joined
    assert "ignored" in joined.lower() or "deprecat" in joined.lower()


def test_package_json_output_is_clean_with_deprecated_flag(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # A deprecated flag must NOT print a warning to stdout ahead of the JSON payload.
    import json as _json

    (tmp_path / "estimator.py").write_text(_EST, encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    out = tmp_path.parent / "j.tar.gz"

    rc = cli.main(
        [
            "package",
            "--estimator",
            str(tmp_path / "estimator.py"),
            "--requirements",
            str(tmp_path / "requirements.txt"),
            "--output",
            str(out),
            "--json",
        ]
    )

    assert rc == 0
    # stdout must be valid JSON — no warning text prepended.
    _json.loads(capsys.readouterr().out)


class _TTYStdin:
    def isatty(self) -> bool:
        return True


def test_package_folder_confirm_decline_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "estimator.py").write_text(_EST, encoding="utf-8")
    out = tmp_path.parent / "declined.tar.gz"
    monkeypatch.setattr("sys.stdin", _TTYStdin())
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")
    captured = _spy(monkeypatch)

    rc = cli.main(["package", "--estimator", str(tmp_path), "--output", str(out)])

    assert rc == 0
    assert not out.exists()
    assert "Aborted" in "\n".join(captured)


def test_package_folder_confirm_accept_writes_archive(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "estimator.py").write_text(_EST, encoding="utf-8")
    out = tmp_path.parent / "accepted.tar.gz"
    monkeypatch.setattr("sys.stdin", _TTYStdin())
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "y")

    rc = cli.main(["package", "--estimator", str(tmp_path), "--output", str(out)])

    assert rc == 0
    assert out.exists()


def test_submit_estimator_folder_shows_file_list_before_upload(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "estimator.py").write_text(_EST, encoding="utf-8")
    (tmp_path / "helper.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    # submit has no --output; package_submission defaults to cwd. chdir so the
    # generated artifact lands in tmp_path (auto-cleaned), not the repo root.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    monkeypatch.setattr(cli, "AIcrowdClient", _FakeClient, raising=False)
    captured = _spy(monkeypatch)

    rc = cli.main(["submit", "--estimator", str(tmp_path), "--yes"])

    assert rc == 0
    assert "helper.py" in "\n".join(captured)
