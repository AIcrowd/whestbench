"""Regression tests for the import-isolation fixture in conftest.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_dataset_file_subset_runs_standalone():
    """A single-file pytest run must survive ``_isolate_import_state``.

    Repro for a real failure mode: in a fresh pytest process, the first
    in-process test imports ``whestbench.cli`` → ``flopscope`` → ``numpy``;
    the fixture then evicted numpy from ``sys.modules``, and the next test's
    re-import died with ``ImportError: cannot load module more than once per
    process`` (numpy 2.x refuses to re-initialise its C extension). Full-suite
    runs masked this because other test modules import numpy at collection
    time, putting it in every snapshot the fixture restores from.
    """
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_cli_dataset.py",
            "-k",
            "download_cache_hit or download_materialize",
            "-q",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    combined = res.stdout + res.stderr
    assert "cannot load module more than once" not in combined, (
        f"import-isolation fixture evicted a one-init-per-process C extension:\n{combined}"
    )
    assert res.returncode == 0, f"single-file subset run failed (rc={res.returncode}):\n{combined}"


def test_one_init_retention_is_limited_to_installed_packages(tmp_path: Path):
    """Only the *installed* one-init packages are exempt from eviction.

    The estimator loader puts the submission dir at ``sys.path[0]``, so a
    submission shipping a sibling named ``numpy.py`` or ``datasets.py`` can
    get imported under a whitelisted top-level name. Such an impostor never
    initialised the real C extension, so it must still be evicted — retaining
    it would shadow the real package for every later test.
    """
    import importlib.machinery
    import types

    import conftest as _conftest
    import numpy

    # The real installed package is retained.
    assert _conftest._keep_during_eviction("numpy", {"numpy": numpy})
    assert _conftest._keep_during_eviction("numpy.linalg", {"numpy": numpy})

    # A same-named impostor from a submission dir is NOT retained.
    fake_path = tmp_path / "numpy.py"
    fake_path.write_text("")
    fake = types.ModuleType("numpy")
    fake.__spec__ = importlib.machinery.ModuleSpec("numpy", None, origin=str(fake_path))
    assert not _conftest._keep_during_eviction("numpy", {"numpy": fake})

    # No spec/origin (e.g. a bare types.ModuleType stub) is NOT retained.
    bare = types.ModuleType("datasets")
    assert not _conftest._keep_during_eviction("datasets", {"datasets": bare})

    # Non-whitelisted names are never retained, installed or not.
    assert not _conftest._keep_during_eviction("rich", {"numpy": numpy})
