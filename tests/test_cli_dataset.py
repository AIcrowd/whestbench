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
        "--output",
        str(out),
        check=True,
    )
    res = _run_whest("dataset", "inspect", str(out))
    assert res.returncode == 0
    assert "3.0" in res.stdout
    assert "flopscope" in res.stdout
    # Under seed_protocol 3.0, there is no `seed` field in metadata.
    assert "n_mlps" in res.stdout


def test_whest_dataset_combine_splits_help_lists_subcommand():
    res = _run_whest("dataset", "--help")
    assert res.returncode == 0
    assert "combine-splits" in res.stdout


def test_whest_dataset_combine_splits_produces_multi_split_dir(tmp_path: Path):
    pub = tmp_path / "pub"
    hold = tmp_path / "hold"
    for split, out_dir in (("public", pub), ("holdout", hold)):
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
            "--split",
            split,
            "--output",
            str(out_dir),
        )
        assert res.returncode == 0, res.stderr

    out = tmp_path / "combined"
    res = _run_whest(
        "dataset",
        "combine-splits",
        str(pub),
        str(hold),
        "--output",
        str(out),
    )
    assert res.returncode == 0, res.stderr
    assert (out / "data" / "public-00000-of-00001.parquet").is_file()
    assert (out / "data" / "holdout-00000-of-00001.parquet").is_file()
    md = json.loads((out / "metadata.json").read_text())
    assert set(md["splits"].keys()) == {"public", "holdout"}


def test_whest_dataset_combine_splits_rejects_existing_output(tmp_path: Path):
    pub = tmp_path / "pub"
    hold = tmp_path / "hold"
    for split, out_dir in (("public", pub), ("holdout", hold)):
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
            "--split",
            split,
            "--output",
            str(out_dir),
        )
        assert res.returncode == 0, res.stderr

    out = tmp_path / "combined"
    out.mkdir()
    res = _run_whest(
        "dataset",
        "combine-splits",
        str(pub),
        str(hold),
        "--output",
        str(out),
    )
    assert res.returncode != 0


def test_whest_dataset_inspect_multi_split_output(tmp_path: Path):
    pub = tmp_path / "pub"
    hold = tmp_path / "hold"
    for split, out_dir in (("public", pub), ("holdout", hold)):
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
            "--split",
            split,
            "--output",
            str(out_dir),
        )
    out = tmp_path / "combined"
    _run_whest("dataset", "combine-splits", str(pub), str(hold), "--output", str(out))

    res = _run_whest("dataset", "inspect", str(out))
    assert res.returncode == 0, res.stderr
    # Multi-split inspect must mention each split name AND the multi-split marker.
    assert "public" in res.stdout
    assert "holdout" in res.stdout
    assert "multi-split" in res.stdout.lower()


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
        "--split",
        "my_split",
        "--output",
        str(out),
    )
    assert res.returncode != 0
    combined = (res.stderr + res.stdout).lower()
    assert "[a-z][a-z0-9]" in combined or "convention" in combined


def test_whest_dataset_pull_accepts_split_flag(tmp_path: Path):
    """Smoke test: pull --split SPLIT is accepted (downloads only that split's parquet).

    Skipped on no network — pulls from the public smoke-test repo.
    """
    res = _run_whest(
        "dataset",
        "pull",
        "aicrowd/arc-whestbench-2026-smoke-test",
        "--split",
        "public",
        "--output",
        str(tmp_path / "pulled"),
    )
    # Tolerate network/offline failure — the test only enforces that --split is
    # an accepted argument, not that the pull itself always succeeds.
    if res.returncode == 0:
        assert (tmp_path / "pulled" / "data" / "public-00000-of-00001.parquet").is_file()
    else:
        unrecognized = "unrecognized arguments" in res.stderr.lower() and "--split" in res.stderr
        assert not unrecognized, f"--split must be an accepted flag; got: {res.stderr}"


def test_whest_dataset_pull_rejects_nonexistent_split(tmp_path: Path):
    """pull --split <typo> should error rather than silently producing an empty dir."""
    res = _run_whest(
        "dataset",
        "pull",
        "aicrowd/arc-whestbench-2026-smoke-test",
        "--split",
        "nonexistent-split",
        "--output",
        str(tmp_path / "pulled"),
    )
    # Network might fail — tolerate that, but if we DID reach the validation
    # logic, the exit must be nonzero AND the error must mention "matched no parquet".
    combined = res.stderr + res.stdout
    if "matched no parquet" in combined.lower() or res.returncode != 0:
        # Either we hit the validation OR the network failed cleanly.
        pass
    else:
        # Returncode 0 with no error → silent empty dir, the bug we're guarding against.
        assert False, (
            f"pull with bogus split should not silently succeed; got returncode={res.returncode}, "
            f"output:\n{res.stdout}\n{res.stderr}"
        )


def test_whest_dataset_bake_accepts_mlp_seeds_file(tmp_path: Path):
    """`whest dataset bake --mlp-seeds FILE.json` reads + bakes with explicit seeds."""
    seeds_file = tmp_path / "seeds.json"
    seeds_file.write_text(json.dumps([42, 99, 1234, 5678]))
    out = tmp_path / "ds"
    res = _run_whest(
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
        "--mlp-seeds",
        str(seeds_file),
        "--output",
        str(out),
    )
    assert res.returncode == 0, res.stderr
    md = json.loads((out / "metadata.json").read_text())
    assert md["seed_protocol"]["name"] == "whestbench_explicit_per_mlp_seeds"
    assert "seed" not in md


def test_whest_dataset_bake_auto_generates_when_no_seed_flag(tmp_path: Path):
    """No --seed and no --mlp-seeds → auto-generate."""
    out = tmp_path / "ds"
    res = _run_whest(
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
        "--output",
        str(out),
    )
    assert res.returncode == 0, res.stderr
    md = json.loads((out / "metadata.json").read_text())
    assert md["seed_protocol"]["version"] == "3.0"


def test_whest_dataset_bake_rejects_legacy_seed_flag(tmp_path: Path):
    """`--seed N` produces a clear migration error."""
    out = tmp_path / "ds"
    res = _run_whest(
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
        "42",
        "--output",
        str(out),
    )
    assert res.returncode != 0
    combined = (res.stderr + res.stdout).lower()
    assert "mlp-seeds" in combined or "mlp_seeds" in combined
    assert "no longer" in combined or "deprecated" in combined or "not supported" in combined


def test_whest_dataset_bake_rejects_mlp_seeds_wrong_length(tmp_path: Path):
    """File length must match --n-mlps."""
    seeds_file = tmp_path / "seeds.json"
    seeds_file.write_text(json.dumps([1, 2, 3]))  # length 3
    out = tmp_path / "ds"
    res = _run_whest(
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
        "--mlp-seeds",
        str(seeds_file),
        "--output",
        str(out),
    )
    assert res.returncode != 0
    combined = (res.stderr + res.stdout).lower()
    assert "length" in combined and "n_mlps" in combined


def test_whest_dataset_bake_rejects_mlp_seeds_malformed_file(tmp_path: Path):
    """File must parse as a JSON array."""
    seeds_file = tmp_path / "seeds.json"
    seeds_file.write_text("not valid json {")
    out = tmp_path / "ds"
    res = _run_whest(
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
        "--mlp-seeds",
        str(seeds_file),
        "--output",
        str(out),
    )
    assert res.returncode != 0
    combined = (res.stderr + res.stdout).lower()
    assert "json" in combined or "parse" in combined or "invalid" in combined


def test_whest_dataset_inspect_v3_mentions_protocol(tmp_path: Path):
    """inspect output for a 3.0 dataset should mention the protocol."""
    out = tmp_path / "ds"
    res = _run_whest(
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
        "--output",
        str(out),
    )
    assert res.returncode == 0, res.stderr
    res = _run_whest("dataset", "inspect", str(out))
    assert res.returncode == 0, res.stderr
    assert "whestbench_explicit_per_mlp_seeds" in res.stdout or "3.0" in res.stdout
