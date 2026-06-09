from pathlib import Path

import pytest

from whestbench import limits
from whestbench.packaging import (
    collect_submission_files,
    enforce_submission_caps,
    summarize_submission,
)


def test_collect_includes_py_and_data_excludes_junk(tmp_path: Path):
    (tmp_path / "estimator.py").write_text("x = 1\n")
    (tmp_path / "helper.py").write_text("y = 2\n")
    (tmp_path / "weights.npz").write_bytes(b"\x00" * 10)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "c.pyc").write_bytes(b"\x00")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "x").write_text("big")
    (tmp_path / "old.tar.gz").write_bytes(b"\x00")
    names = {p.name for p in collect_submission_files(tmp_path)}
    assert names == {"estimator.py", "helper.py", "weights.npz"}


def test_collect_honors_whestignore(tmp_path: Path):
    (tmp_path / "estimator.py").write_text("x = 1\n")
    (tmp_path / "scratch.npz").write_bytes(b"\x00")
    (tmp_path / ".whestignore").write_text("scratch.npz\n")
    names = {p.name for p in collect_submission_files(tmp_path)}
    assert "scratch.npz" not in names
    assert ".whestignore" not in names


def test_caps_ok(tmp_path):
    files = []
    for i in range(3):
        f = tmp_path / f"f{i}.npy"
        f.write_bytes(b"\x00" * 10)
        files.append(f)
    enforce_submission_caps(files)  # no raise


def test_too_many_files(tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "MAX_SUBMISSION_FILES", 2)
    files = [tmp_path / f"f{i}.npy" for i in range(3)]
    for f in files:
        f.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="files"):
        enforce_submission_caps(files)


def test_too_large(tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "MAX_SUBMISSION_BYTES", 5)
    f = tmp_path / "big.npy"
    f.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="MB|bytes|size"):
        enforce_submission_caps([f])


def test_summary_reports_total_count_and_unreachable(tmp_path):
    (tmp_path / "estimator.py").write_text("from helper import f\n")
    (tmp_path / "helper.py").write_text("def f():\n    return 0\n")
    (tmp_path / "orphan.py").write_text("# never imported\n")
    (tmp_path / "weights.npz").write_bytes(b"\x00" * 8)
    s = summarize_submission(tmp_path / "estimator.py")
    assert s.file_count == 4
    assert s.total_bytes > 0
    assert "orphan.py" in s.unreachable_py
    assert "helper.py" not in s.unreachable_py
