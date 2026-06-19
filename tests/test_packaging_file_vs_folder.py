"""`--estimator FILE` ships only that file; `--estimator DIR` ships the folder."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from whestbench.packaging import package_submission, summarize_submission

_SELF_CONTAINED = (
    "from whestbench import BaseEstimator\n"
    "class Estimator(BaseEstimator):\n"
    "    def predict(self, mlp, budget):\n"
    "        import flopscope.numpy as fnp\n"
    "        return fnp.zeros((mlp.depth, mlp.width))\n"
)


def _folder_with_siblings(tmp_path: Path) -> None:
    (tmp_path / "estimator.py").write_text(_SELF_CONTAINED, encoding="utf-8")
    (tmp_path / "helper.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    (tmp_path / "weights.npz").write_bytes(b"\x00" * 16)


def _archive_names(out: Path) -> set[str]:
    with tarfile.open(out, "r:gz") as tf:
        return set(tf.getnames())


def test_file_arg_ships_only_that_file(tmp_path: Path) -> None:
    _folder_with_siblings(tmp_path)
    out = tmp_path / "submission.tar.gz"

    package_submission(tmp_path / "estimator.py", output_path=out)

    assert _archive_names(out) == {"estimator.py", "manifest.json"}


def test_dir_arg_ships_whole_folder(tmp_path: Path) -> None:
    _folder_with_siblings(tmp_path)
    out = tmp_path.parent / "submission.tar.gz"

    package_submission(tmp_path, output_path=out)

    assert {"estimator.py", "helper.py", "weights.npz", "manifest.json"} <= _archive_names(out)


def test_dir_arg_requires_estimator_py(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text(_SELF_CONTAINED, encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="estimator.py"):
        package_submission(tmp_path, output_path=tmp_path.parent / "submission.tar.gz")


def test_summary_mode_is_file_for_file_arg(tmp_path: Path) -> None:
    _folder_with_siblings(tmp_path)

    summary = summarize_submission(tmp_path / "estimator.py")

    assert summary.mode == "file"
    assert summary.file_count == 1


def test_summary_mode_is_folder_for_dir_arg(tmp_path: Path) -> None:
    _folder_with_siblings(tmp_path)

    summary = summarize_submission(tmp_path)

    assert summary.mode == "folder"
    assert summary.file_count == 3


def test_file_arg_must_be_named_estimator_py(tmp_path: Path) -> None:
    # The manifest entrypoint module is hardcoded "estimator"; a single file with
    # any other name would package locally but be rejected by the grader. Fail fast.
    (tmp_path / "my_solution.py").write_text(_SELF_CONTAINED, encoding="utf-8")

    with pytest.raises(ValueError, match="estimator.py"):
        package_submission(tmp_path / "my_solution.py", output_path=tmp_path.parent / "x.tar.gz")


def test_folder_mode_errors_if_estimator_excluded_by_whestignore(tmp_path: Path) -> None:
    # A .whestignore/.gitignore pattern matching estimator.py would silently drop the
    # entrypoint from the archive while the manifest still declares it. Refuse loudly.
    _folder_with_siblings(tmp_path)
    (tmp_path / ".whestignore").write_text("estimator.py\n", encoding="utf-8")

    with pytest.raises(ValueError, match="estimator.py"):
        package_submission(tmp_path, output_path=tmp_path.parent / "x.tar.gz")
