"""Dataset persists mlp.seed and uses SEED_PROTOCOL_VERSION 3.0 by default."""

from __future__ import annotations

import pytest

from whestbench.dataset import (
    create_dataset,
    iter_mlps,
    load_dataset,
)
from whestbench.dataset_io import (
    SEED_PROTOCOL_VERSION,
    SEED_PROTOCOL_VERSION_V3,
    InvalidDatasetError,
)


def test_seed_protocol_version_is_2():
    # SEED_PROTOCOL_VERSION (v2.0) constant is still exported for back-compat.
    assert SEED_PROTOCOL_VERSION == "2.0"


def test_seed_protocol_version_v3_is_3():
    assert SEED_PROTOCOL_VERSION_V3 == "3.0"


def test_dataset_roundtrip_persists_estimator_seeds(tmp_path):
    """Round-trip: create_dataset under v3; iter_mlps derives estimator seeds."""
    out = tmp_path / "test"
    mlp_seeds = [42000, 42001, 42002]
    create_dataset(
        n_mlps=3,
        n_samples=100,
        width=8,
        depth=2,
        mlp_seeds=mlp_seeds,
        output_path=out,
    )
    ds = load_dataset(out, split="public")
    seeds = [m.seed for m in iter_mlps(ds)]
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == 3, f"seeds not distinct: {seeds}"


def test_dataset_create_reproduces_same_seeds(tmp_path):
    """Same mlp_seeds → same per-MLP estimator seeds across two dataset bakes."""
    mlp_seeds = [99000, 99001, 99002]
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    for path in (out1, out2):
        create_dataset(
            n_mlps=3,
            n_samples=100,
            width=8,
            depth=2,
            mlp_seeds=mlp_seeds,
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
