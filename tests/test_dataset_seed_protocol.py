"""Dataset persists mlp.seed and uses SEED_PROTOCOL_VERSION 2.0."""

from __future__ import annotations

import pytest

from whestbench.dataset import (
    create_dataset,
    iter_mlps,
    load_dataset,
)
from whestbench.dataset_io import (
    SEED_PROTOCOL_VERSION,
    InvalidDatasetError,
)


def test_seed_protocol_version_is_2():
    assert SEED_PROTOCOL_VERSION == "2.0"


def test_dataset_roundtrip_persists_mlp_seeds(tmp_path):
    """Round-trip: create_dataset writes mlp.seed; load_dataset restores them."""
    out = tmp_path / "test"
    create_dataset(
        n_mlps=3,
        n_samples=100,
        width=8,
        depth=2,
        seed=42,
        output_path=out,
    )
    ds = load_dataset(out, split="public")
    seeds = [m.seed for m in iter_mlps(ds)]
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == 3, f"seeds not distinct: {seeds}"


def test_dataset_create_reproduces_same_seeds(tmp_path):
    """Same spec.seed → same per-MLP seeds across two dataset bakes."""
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    for path in (out1, out2):
        create_dataset(
            n_mlps=3,
            n_samples=100,
            width=8,
            depth=2,
            seed=99,
            output_path=path,
        )
    seeds_a = [m.seed for m in iter_mlps(load_dataset(out1, split="public"))]
    seeds_b = [m.seed for m in iter_mlps(load_dataset(out2, split="public"))]
    assert seeds_a == seeds_b


def test_loading_old_schema_version_fails_with_clear_message(tmp_path):
    """Datasets with old schema_version must fail to load with a clear error."""
    import json

    dataset_dir = tmp_path / "v2_dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "2.4",
                "seed_protocol": {
                    "name": "whestbench_seedsequence_hierarchy",
                    "version": "2.0",
                    "seeded": True,
                },
                "created_at_utc": "2026-05-14T00:00:00+00:00",
                "seed": 1,
                "n_mlps": 1,
                "n_samples": 1,
                "width": 4,
                "depth": 2,
                "hardware": {},
            }
        )
    )

    with pytest.raises(InvalidDatasetError, match="schema_version"):
        load_dataset(dataset_dir)


def test_loading_dataset_with_wrong_seed_protocol_version_fails(tmp_path):
    """Datasets with mismatched seed_protocol.version must fail to load."""
    import json

    dataset_dir = tmp_path / "v1_sp"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
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
                "hardware": {},
            }
        )
    )

    with pytest.raises(InvalidDatasetError, match="seed_protocol"):
        load_dataset(dataset_dir)
