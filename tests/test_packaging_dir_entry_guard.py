"""A manifest `files[]` entry must be a regular file — never a bare directory.

Regression for whestbench#107: several prod submissions carried a hand-rolled
`manifest.json` whose `files[]` listed a bundled subpackage as a *directory*
entry (e.g. ``{"name": "arc_tools/", "sha256": "..."}``). The grader hashes each
entry with ``sha256(path.read_bytes())``, which raises ``IsADirectoryError`` on a
directory and crashed every eval worker ~1s in, with no participant feedback.

whestbench's own ``whest package`` never produces such a manifest — folder mode
lists the *individual* files of a subpackage, and ``_sha256`` cannot hash a
directory at all (it would crash at package time). These tests lock both halves:

1. The hardening: ``build_manifest`` / ``_sha256`` fail **loudly** (clear
   ``ValueError``) on a non-file entry, instead of a cryptic ``IsADirectoryError``,
   so a directory can never silently become — or crash on — a ``files[]`` entry.
2. The supported path: a folder submission containing a subpackage directory
   yields manifest entries for the individual files, never a bare-dir entry.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path

import pytest

from whestbench.packaging import _sha256, build_manifest, package_submission

_SELF_CONTAINED = (
    "from whestbench import BaseEstimator\n"
    "class Estimator(BaseEstimator):\n"
    "    def predict(self, mlp, budget):\n"
    "        import flopscope.numpy as fnp\n"
    "        return fnp.zeros((mlp.depth, mlp.width))\n"
)


def test_sha256_rejects_directory(tmp_path: Path) -> None:
    # The lowest layer: hashing a directory must fail with an actionable ValueError,
    # not the bare ``IsADirectoryError`` that took down the grader.
    d = tmp_path / "arc_tools"
    d.mkdir()
    with pytest.raises(ValueError, match="regular file|directory"):
        _sha256(d)


def test_build_manifest_rejects_directory_entry(tmp_path: Path) -> None:
    # Reproduces the exact shape of the prod manifest: a directory in files[].
    # Before the guard this raised IsADirectoryError (crashing the grader); now it
    # must raise a clear ValueError that names the offending entry.
    (tmp_path / "estimator.py").write_text(_SELF_CONTAINED, encoding="utf-8")
    subpkg = tmp_path / "arc_tools"
    subpkg.mkdir()
    (subpkg / "__init__.py").write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="arc_tools") as exc:
        build_manifest(
            class_name="Estimator",
            root=tmp_path,
            files=[tmp_path / "estimator.py", subpkg],
        )
    # The message must be actionable, not a stack-trace-only failure.
    assert "regular file" in str(exc.value) or "directory" in str(exc.value)


def test_folder_submission_with_subpackage_lists_individual_files(tmp_path: Path) -> None:
    # The SUPPORTED way to ship a helper subpackage: folder mode lists each file of
    # the package, never a bare ``arc_tools/`` directory entry.
    (tmp_path / "estimator.py").write_text(_SELF_CONTAINED, encoding="utf-8")
    pkg = tmp_path / "arc_tools"
    (pkg / "polynomial").mkdir(parents=True)
    (pkg / "__init__.py").write_text("\n", encoding="utf-8")
    (pkg / "_arc_mlp.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    (pkg / "polynomial" / "__init__.py").write_text("\n", encoding="utf-8")
    out = tmp_path.parent / "submission.tar.gz"

    package_submission(tmp_path, output_path=out)

    with tarfile.open(out, "r:gz") as tf:
        manifest = json.loads(tf.extractfile("manifest.json").read())  # type: ignore[union-attr]
        by_name = {f["name"]: f["sha256"] for f in manifest["files"]}
        # The supported path must yield a *gradeable* manifest: the grader re-hashes
        # every file, so each entry's sha256 must be a real digest of the bundled bytes.
        # Prove it for a file nested INSIDE the subpackage subdirectory — the exact case
        # the bad hand-rolled manifests got wrong by listing a bare, unhashable directory.
        nested = tf.extractfile("arc_tools/_arc_mlp.py").read()  # type: ignore[union-attr]
        assert by_name["arc_tools/_arc_mlp.py"] == hashlib.sha256(nested).hexdigest()
    names = set(by_name)

    # Every entry carries a real 64-hex digest — never a directory placeholder.
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in by_name.values())

    # Individual files are listed...
    assert {
        "estimator.py",
        "arc_tools/__init__.py",
        "arc_tools/_arc_mlp.py",
        "arc_tools/polynomial/__init__.py",
    } <= names
    # ...and no bare-directory entry is present (the thing that crashed the grader).
    assert "arc_tools" not in names
    assert "arc_tools/" not in names
    assert not any(n.endswith("/") for n in names)
