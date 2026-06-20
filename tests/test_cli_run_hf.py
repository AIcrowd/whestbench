"""Tests for `whest run --dataset` URL resolution."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from whestbench.cli import _resolve_dataset_arg, _warn_if_hf_dataset_unpinned


def _run_whest(*args, cwd=None, check=False):
    return subprocess.run(
        ["uv", "run", "whest", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=check,
    )


def test_resolves_local_relative_path():
    repo, rev, is_local = _resolve_dataset_arg("./eval-v1", revision=None)
    assert is_local is True
    assert repo == "./eval-v1"


def test_resolves_local_absolute_path():
    repo, rev, is_local = _resolve_dataset_arg("/tmp/x", revision=None)
    assert is_local is True


def test_resolves_hf_url_scheme():
    repo, rev, is_local = _resolve_dataset_arg("hf://aicrowd/arc-whestbench-2026@v1", revision=None)
    assert is_local is False
    assert repo == "aicrowd/arc-whestbench-2026"
    assert rev == "v1"


def test_resolves_hf_url_without_tag():
    repo, rev, is_local = _resolve_dataset_arg("hf://aicrowd/arc-whestbench-2026", revision=None)
    assert is_local is False
    assert repo == "aicrowd/arc-whestbench-2026"
    assert rev is None


def test_hf_url_without_tag_honors_revision_flag():
    """Regression: `hf://owner/repo` (no @rev) must honor --revision.

    Previously this branch returned None unconditionally, silently dropping
    --revision so the run loaded the default branch (`main`) instead of the
    pinned tag.
    """
    repo, rev, is_local = _resolve_dataset_arg("hf://aicrowd/arc-whestbench-2026", revision="v1")
    assert is_local is False
    assert repo == "aicrowd/arc-whestbench-2026"
    assert rev == "v1"


def test_hf_embedded_rev_takes_precedence_over_revision_flag():
    """An embedded @rev pins explicitly and wins over a conflicting --revision."""
    repo, rev, is_local = _resolve_dataset_arg("hf://aicrowd/arc-whestbench-2026@v1", revision="v2")
    assert is_local is False
    assert repo == "aicrowd/arc-whestbench-2026"
    assert rev == "v1"


def test_resolves_repo_with_revision_flag():
    repo, rev, is_local = _resolve_dataset_arg("aicrowd/arc-whestbench-2026", revision="v1")
    assert is_local is False
    assert repo == "aicrowd/arc-whestbench-2026"
    assert rev == "v1"


def test_rejects_bare_repo_without_revision_or_prefix():
    with pytest.raises(SystemExit, match="hf://"):
        _resolve_dataset_arg("aicrowd/arc-whestbench-2026", revision=None)


def test_warns_on_stderr_when_hf_dataset_unpinned(capsys):
    """An unpinned hf:// load must announce the main fallback (no silent main)."""
    _warn_if_hf_dataset_unpinned("aicrowd/arc-whestbench-2026", None)
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing on stdout — must not corrupt --json output
    assert "no revision pinned" in captured.err
    assert "main" in captured.err
    assert "--revision" in captured.err  # actionable: tells the user how to pin


def test_no_warning_when_hf_dataset_pinned(capsys):
    """A pinned revision (tag or @rev) is the happy path — stay silent."""
    _warn_if_hf_dataset_unpinned("aicrowd/arc-whestbench-2026", "v1")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_whest_run_accepts_split_flag_on_dataset(tmp_path: Path):
    """`whest run --dataset hf://... --split <name>` is accepted as a CLI flag."""
    estimator = tmp_path / "noop_estimator.py"
    estimator.write_text(
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        import flopscope.numpy as fnp\n"
        "        return fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)\n"
    )
    res = _run_whest(
        "run",
        "--estimator",
        str(estimator),
        "--dataset",
        "aicrowd/arc-whestbench-2026-smoke-test",
        "--split",
        "public",
        "--n-mlps",
        "1",
        "--flop-budget",
        "100000",
        "--output",
        str(tmp_path / "out.json"),
    )
    # Either succeeds end-to-end, or fails for non-arg reasons (network, etc.).
    # The only failure mode we explicitly fail on is "--split not recognised".
    if res.returncode != 0:
        # We explicitly fail only if argparse rejected --split as unknown.
        # Python argparse emits "unrecognized arguments: --split" for unknown flags.
        unrecognized = "unrecognized arguments" in res.stderr.lower() and "--split" in res.stderr
        assert not unrecognized, f"--split must be an accepted flag; got: {res.stderr}"
