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
