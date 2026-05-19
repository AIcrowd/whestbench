"""Dataset bundle persists mlp.seed and bumps SEED_PROTOCOL_VERSION to 2.0."""

from __future__ import annotations

import json

import numpy as np
import pytest

from whestbench.dataset import (
    SEED_PROTOCOL_VERSION,
    create_dataset,
    load_dataset,
)


def test_seed_protocol_version_is_2():
    assert SEED_PROTOCOL_VERSION == "2.0"


def test_dataset_roundtrip_persists_mlp_seeds(tmp_path):
    """Round-trip: create_dataset writes mlp.seed; load_dataset restores them."""
    out = tmp_path / "test.npz"
    create_dataset(
        n_mlps=3,
        n_samples=100,
        width=8,
        depth=2,
        seed=42,
        output_path=out,
    )
    bundle = load_dataset(out)
    seeds = [m.seed for m in bundle.mlps]
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == 3, f"seeds not distinct: {seeds}"


def test_dataset_create_reproduces_same_seeds(tmp_path):
    """Same spec.seed → same per-MLP seeds across two dataset bakes."""
    out1 = tmp_path / "a.npz"
    out2 = tmp_path / "b.npz"
    for path in (out1, out2):
        create_dataset(
            n_mlps=3,
            n_samples=100,
            width=8,
            depth=2,
            seed=99,
            output_path=path,
        )
    seeds_a = [m.seed for m in load_dataset(out1).mlps]
    seeds_b = [m.seed for m in load_dataset(out2).mlps]
    assert seeds_a == seeds_b


def test_loading_v1_0_dataset_fails_with_clear_message(tmp_path):
    """Datasets baked with SEED_PROTOCOL_VERSION=1.0 must fail to load with a clear error."""
    # Synthesize a minimal v1.0-looking .npz by writing the metadata directly.
    out = tmp_path / "v1.npz"
    metadata = {
        "schema_version": "2.1",
        "seed_protocol": {
            "name": "whestbench_seedsequence_hierarchy",
            "version": "1.0",
            "seeded": True,
        },
        "created_at_utc": "2026-05-14T00:00:00+00:00",
        "seed": 1,
        "n_mlps": 1,
        "n_samples": 1,
        "width": 4,
        "depth": 2,
        "flop_budget": 1000000,
        "hardware": {},
    }
    weights = np.zeros((1, 2, 4, 4), dtype=np.float32)
    all_means = np.zeros((1, 2, 4), dtype=np.float32)
    final_means = np.zeros((1, 4), dtype=np.float32)
    avg_variances = np.array([0.0], dtype=np.float64)

    np.savez(
        out,
        metadata=np.array(json.dumps(metadata)),
        weights=weights,
        all_layer_means=all_means,
        final_means=final_means,
        avg_variances=avg_variances,
        sampling_budget_breakdowns=np.array(json.dumps([])),
    )

    with pytest.raises(ValueError, match=r"seed_protocol.*1\.0.*2\.0"):
        load_dataset(out)
