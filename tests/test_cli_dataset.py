"""Tests for the new `whest dataset` subcommand group."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run_whest(*args, cwd=None, check=False):
    return subprocess.run(
        ["uv", "run", "whest", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=check,
    )


def test_whest_dataset_help_lists_subcommands():
    res = _run_whest("dataset", "--help")
    assert res.returncode == 0
    for sub in ("bake", "push", "pull", "merge", "inspect"):
        assert sub in res.stdout


def test_whest_dataset_bake_outputs_three_files(tmp_path: Path):
    out = tmp_path / "ds"
    res = _run_whest(
        "dataset",
        "bake",
        "--n-mlps",
        "2",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--output",
        str(out),
    )
    assert res.returncode == 0, res.stderr
    assert (out / "data" / "public-00000-of-00001.parquet").is_file()
    assert (out / "metadata.json").is_file()
    assert (out / "README.md").is_file()


def test_whest_dataset_bake_with_holdout_split(tmp_path: Path):
    out = tmp_path / "ds"
    res = _run_whest(
        "dataset",
        "bake",
        "--n-mlps",
        "2",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--split",
        "holdout",
        "--output",
        str(out),
    )
    assert res.returncode == 0, res.stderr
    assert (out / "data" / "holdout-00000-of-00001.parquet").is_file()


def test_whest_dataset_bake_with_slice(tmp_path: Path):
    out = tmp_path / "ds"
    res = _run_whest(
        "dataset",
        "bake",
        "--n-mlps",
        "8",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--slice",
        "0/2",
        "--output",
        str(out),
    )
    assert res.returncode == 0, res.stderr
    md = json.loads((out / "metadata.json").read_text())
    assert md["is_partial"] is True
    assert md["mlp_range"] == [0, 4]
    assert md["total_n_mlps"] == 8


def test_whest_dataset_bake_with_mlp_range(tmp_path: Path):
    out = tmp_path / "ds"
    res = _run_whest(
        "dataset",
        "bake",
        "--n-mlps",
        "10",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--mlp-range",
        "2-5",
        "--output",
        str(out),
    )
    # CLI form is inclusive-inclusive: 2-5 means MLPs 2..5 (4 MLPs)
    assert res.returncode == 0, res.stderr
    md = json.loads((out / "metadata.json").read_text())
    assert md["mlp_range"] == [2, 6]  # Python form exclusive-end
    assert md["n_mlps"] == 4


def test_whest_dataset_merge_combines_partials(tmp_path: Path):
    p0 = tmp_path / "p0"
    p1 = tmp_path / "p1"
    _run_whest(
        "dataset",
        "bake",
        "--n-mlps",
        "4",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--mlp-range",
        "0-1",
        "--output",
        str(p0),
        check=True,
    )
    _run_whest(
        "dataset",
        "bake",
        "--n-mlps",
        "4",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--mlp-range",
        "2-3",
        "--output",
        str(p1),
        check=True,
    )
    merged = tmp_path / "merged"
    res = _run_whest("dataset", "merge", str(p0), str(p1), "--output", str(merged))
    assert res.returncode == 0, res.stderr
    md = json.loads((merged / "metadata.json").read_text())
    assert md["n_mlps"] == 4
    assert "is_partial" not in md


def test_whest_dataset_inspect_prints_metadata(tmp_path: Path):
    out = tmp_path / "ds"
    _run_whest(
        "dataset",
        "bake",
        "--n-mlps",
        "2",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--output",
        str(out),
        check=True,
    )
    res = _run_whest("dataset", "inspect", str(out))
    assert res.returncode == 0
    assert "3.0" in res.stdout
    assert "flopscope" in res.stdout
    assert "seed" in res.stdout.lower()


def test_old_create_dataset_emits_redirect(tmp_path: Path):
    out = tmp_path / "old"
    res = _run_whest(
        "create-dataset",
        "--n-mlps",
        "2",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--output",
        str(out),
    )
    assert res.returncode != 0
    assert "whest dataset bake" in (res.stderr + res.stdout)


def test_whest_dataset_bake_with_arbitrary_split_name(tmp_path: Path):
    out = tmp_path / "ds"
    res = _run_whest(
        "dataset",
        "bake",
        "--n-mlps",
        "2",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--split",
        "my-custom-split",
        "--output",
        str(out),
    )
    assert res.returncode == 0, res.stderr
    assert (out / "data" / "my-custom-split-00000-of-00001.parquet").is_file()


def test_whest_dataset_bake_rejects_uppercase_split(tmp_path: Path):
    out = tmp_path / "ds"
    res = _run_whest(
        "dataset",
        "bake",
        "--n-mlps",
        "2",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--split",
        "Public",
        "--output",
        str(out),
    )
    assert res.returncode != 0
    combined = (res.stderr + res.stdout).lower()
    assert "[a-z][a-z0-9]" in combined or "convention" in combined


def test_whest_dataset_bake_rejects_underscore_split(tmp_path: Path):
    out = tmp_path / "ds"
    res = _run_whest(
        "dataset",
        "bake",
        "--n-mlps",
        "2",
        "--n-samples",
        "100",
        "--width",
        "4",
        "--depth",
        "2",
        "--seed",
        "1",
        "--split",
        "my_split",
        "--output",
        str(out),
    )
    assert res.returncode != 0
    combined = (res.stderr + res.stdout).lower()
    assert "[a-z][a-z0-9]" in combined or "convention" in combined
