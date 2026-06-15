"""Cross-path parity: the local fast (parquet) path and the HF `configs:`
manifest path return identical data for the committed two-config fixture.

This is the coverage that proves our generated dataset-card manifest resolves
to the correct `data_files` — the failure mode that previously only surfaced
against the live HF dataset.
"""

from __future__ import annotations

from pathlib import Path

import datasets as hf_datasets
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "multi_config_v3.0"


def test_parquet_and_manifest_paths_agree():
    from whestbench.dataset import load_dataset

    # Fast path — what whestbench.load_dataset uses for a local dir.
    fast = load_dataset(FIXTURE, split="holdout")
    # HF `configs:` manifest path — the production-remote resolution.
    manifest = hf_datasets.load_dataset(str(FIXTURE), "holdout", split="holdout")

    assert len(fast) == len(manifest) == 2
    assert list(fast["mlp_id"]) == list(manifest["mlp_id"])
    assert list(fast["mlp_seed"]) == list(manifest["mlp_seed"])


def test_manifest_default_config_returns_only_public():
    # name="default" resolves to the public split only (holdout isolation).
    pub = hf_datasets.load_dataset(str(FIXTURE), "default", split="public")
    assert len(pub) == 2
    # The default config does not expose the holdout split.
    with pytest.raises(ValueError, match="Unknown split"):
        hf_datasets.load_dataset(str(FIXTURE), "default", split="holdout")
