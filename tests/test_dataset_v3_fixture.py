"""Regression test for the frozen single-split v3 protocol fixture.

Catches accidental on-disk drift in future PRs.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "single_split_v3_protocol"


def test_v3_fixture_layout():
    assert (FIXTURE / "data" / "public-00000-of-00001.parquet").is_file()
    assert (FIXTURE / "metadata.json").is_file()
    assert (FIXTURE / "README.md").is_file()


def test_v3_fixture_metadata_shape():
    md = json.loads((FIXTURE / "metadata.json").read_text())
    assert md["schema_version"] == "3.0"
    assert md["seed_protocol"]["name"] == "whestbench_explicit_per_mlp_seeds"
    assert md["seed_protocol"]["version"] == "3.0"
    assert "seed" not in md  # no top-level seed under 3.0
    assert md["n_mlps"] == 4
    assert md["width"] == 4
    assert md["depth"] == 2


def test_v3_fixture_loads_via_whestbench():
    from whestbench.dataset import load_dataset
    from whestbench.dataset import metadata as md_accessor

    ds = load_dataset(FIXTURE, split="public")
    assert len(ds) == 4
    md = md_accessor(ds)
    assert md["seed_protocol"]["version"] == "3.0"


def test_v3_fixture_mlp_seeds_in_parquet_are_inputs():
    """Under 3.0, the parquet mlp_seed column IS the input seed list."""
    from whestbench.dataset import load_dataset

    ds = load_dataset(FIXTURE, split="public")
    assert ds["mlp_seed"] == [1001, 2002, 3003, 4004]


def test_v3_fixture_iter_mlps_derives_estimator_seeds():
    """mlp.seed is derived locally from the parquet input seed."""
    import flopscope.numpy as fnp

    from whestbench.dataset import iter_mlps, load_dataset

    ds = load_dataset(FIXTURE, split="public")
    mlps = list(iter_mlps(ds))
    for i, input_seed in enumerate([1001, 2002, 3003, 4004]):
        expected = int(fnp.random.SeedSequence(input_seed).spawn(3)[2].generate_state(1)[0])
        assert mlps[i].seed == expected
        mlps[i].validate()
