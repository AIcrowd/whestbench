"""`--estimator FILE` ships only that file; `--estimator DIR` ships the folder."""

from __future__ import annotations

import json
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


def test_file_arg_renamed_to_estimator_py(tmp_path: Path) -> None:
    # The manifest entrypoint module is hardcoded "estimator", so a single file
    # is always shipped AS estimator.py — whatever the participant named it on
    # disk. (Pre-0.12 this silently produced a tarball the grader rejected with
    # "missing module file: estimator.py"; that was prod bug 312082.)
    src = tmp_path / "01_random.py"
    src.write_text(_SELF_CONTAINED, encoding="utf-8")
    out = tmp_path.parent / "x.tar.gz"

    package_submission(src, output_path=out)

    # Renamed inside the archive; the original name does not appear.
    assert _archive_names(out) == {"estimator.py", "manifest.json"}
    with tarfile.open(out, "r:gz") as tf:
        body = tf.extractfile("estimator.py").read().decode("utf-8")  # type: ignore[union-attr]
        manifest = json.loads(tf.extractfile("manifest.json").read())  # type: ignore[union-attr]
    assert body == _SELF_CONTAINED
    assert manifest["entrypoint"]["module"] == "estimator"
    assert [f["name"] for f in manifest["files"]] == ["estimator.py"]


def test_file_named_estimator_py_unchanged(tmp_path: Path) -> None:
    # The common case (already estimator.py) still ships exactly one file.
    _folder_with_siblings(tmp_path)
    out = tmp_path.parent / "x.tar.gz"

    package_submission(tmp_path / "estimator.py", output_path=out)

    assert _archive_names(out) == {"estimator.py", "manifest.json"}


def test_folder_with_wrong_predict_signature_rejected(tmp_path: Path) -> None:
    # The grader calls predict(mlp, budget). A folder whose Estimator.predict
    # cannot accept those two positional args is rejected at package time
    # (and therefore at submit time) rather than failing per-MLP in the grader.
    bad = (
        "from whestbench import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self):\n"  # missing mlp, budget
        "        return None\n"
    )
    (tmp_path / "estimator.py").write_text(bad, encoding="utf-8")

    with pytest.raises(ValueError, match="predict"):
        package_submission(tmp_path, output_path=tmp_path.parent / "x.tar.gz")


def test_folder_mode_errors_if_estimator_excluded_by_whestignore(tmp_path: Path) -> None:
    # A .whestignore/.gitignore pattern matching estimator.py would silently drop the
    # entrypoint from the archive while the manifest still declares it. Refuse loudly.
    _folder_with_siblings(tmp_path)
    (tmp_path / ".whestignore").write_text("estimator.py\n", encoding="utf-8")

    with pytest.raises(ValueError, match="estimator.py"):
        package_submission(tmp_path, output_path=tmp_path.parent / "x.tar.gz")
