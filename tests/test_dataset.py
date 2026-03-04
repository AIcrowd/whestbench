from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from circuit_estimation.dataset import (
    DatasetBundle,
    create_dataset,
    load_dataset,
)


def test_create_and_load_roundtrip(tmp_path: Path):
    out = tmp_path / "ds.npz"
    create_dataset(
        n_circuits=2,
        n_samples=100,
        width=4,
        max_depth=2,
        budgets=[10, 100],
        seed=42,
        output_path=out,
    )
    assert out.exists()

    bundle = load_dataset(out)
    assert bundle.n_circuits == 2
    assert bundle.metadata["seed"] == 42
    assert bundle.metadata["n_samples"] == 100
    assert bundle.ground_truth_means.shape == (2, 2, 4)
    assert bundle.baseline_times.shape == (2, 2)  # n_budgets x depth
    assert len(bundle.circuits) == 2
    assert bundle.circuits[0].n == 4
    assert bundle.circuits[0].d == 2


def test_create_dataset_auto_seed(tmp_path: Path):
    out = tmp_path / "ds.npz"
    create_dataset(
        n_circuits=1, n_samples=50, width=4, max_depth=1,
        budgets=[10], output_path=out,
    )
    bundle = load_dataset(out)
    assert "seed" in bundle.metadata
    assert isinstance(bundle.metadata["seed"], int)


def test_create_dataset_reproducible(tmp_path: Path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    for path in (a, b):
        create_dataset(
            n_circuits=2, n_samples=100, width=4, max_depth=2,
            budgets=[10], seed=123, output_path=path,
        )
    ba = load_dataset(a)
    bb = load_dataset(b)
    # Circuits are deterministic from the seed — must match exactly.
    for ca, cb in zip(ba.circuits, bb.circuits):
        np.testing.assert_array_equal(ca.gates[0].first, cb.gates[0].first)
        np.testing.assert_array_equal(ca.gates[0].const, cb.gates[0].const)


def test_load_dataset_validates_schema(tmp_path: Path):
    bad_file = tmp_path / "bad.npz"
    np.savez(bad_file, metadata=np.array("{}"))
    with pytest.raises(ValueError, match="schema_version"):
        load_dataset(bad_file)
