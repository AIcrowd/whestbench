"""Provenance fields on metadata.json: whestbench_version, flopscope_version,
bake_config (torch path), cuda_driver_version (CUDA only).

These were added so consumers can reproduce datasets bit-exactly: dataset_torch's
bake math is sensitive to chunk_size + determinism flag state, and CPU bake's
weight init depends on flopscope's RNG (so flopscope_version matters too).

See docs/how-to/parallel-bake.md § "Bit-equivalence requirements" for the
operational context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from whestbench.dataset import create_dataset
from whestbench.dataset_io import validate_metadata


def _read(out: Path) -> dict[str, Any]:
    return json.loads((out / "metadata.json").read_text())


# ----------------- CPU path -----------------


def test_cpu_metadata_has_whestbench_version(tmp_path: Path):
    out = tmp_path / "ds"
    create_dataset(
        n_mlps=2,
        n_samples=1_000,
        width=4,
        depth=2,
        mlp_seeds=[111, 222],
        output_path=out,
    )
    md = _read(out)
    assert "whestbench_version" in md
    assert isinstance(md["whestbench_version"], str)
    # The package is locally installed during tests, so we should never see "unknown"
    assert md["whestbench_version"] != "unknown", (
        "whestbench_version should resolve via importlib.metadata when the "
        "package is installed; got 'unknown'."
    )


def test_cpu_metadata_has_flopscope_version(tmp_path: Path):
    out = tmp_path / "ds"
    create_dataset(
        n_mlps=2,
        n_samples=1_000,
        width=4,
        depth=2,
        mlp_seeds=[1, 2],
        output_path=out,
    )
    md = _read(out)
    assert "flopscope_version" in md
    assert isinstance(md["flopscope_version"], str)


def test_cpu_metadata_has_no_bake_config(tmp_path: Path):
    """bake_config captures torch's determinism flags. CPU/flopscope path has no
    such concept, so the field is absent."""
    out = tmp_path / "ds"
    create_dataset(
        n_mlps=1,
        n_samples=1_000,
        width=4,
        depth=2,
        mlp_seeds=[42],
        output_path=out,
    )
    md = _read(out)
    assert "bake_config" not in md


# ----------------- torch path -----------------


@pytest.fixture
def torch_metadata(tmp_path: Path) -> dict[str, Any]:
    pytest.importorskip("torch")
    from whestbench.dataset_torch import create_dataset_torch

    out = tmp_path / "ds"
    create_dataset_torch(
        n_mlps=2,
        n_samples=1_000,
        width=4,
        depth=2,
        mlp_seeds=[10, 20],
        output_path=out,
        device="cpu",
    )
    return _read(out)


def test_torch_metadata_has_whestbench_and_flopscope_versions(torch_metadata):
    assert "whestbench_version" in torch_metadata
    assert "flopscope_version" in torch_metadata


def test_torch_metadata_has_bake_config(torch_metadata):
    cfg = torch_metadata.get("bake_config")
    assert cfg is not None, "bake_config must be present on the torch path"
    assert isinstance(cfg, dict)
    expected_keys = {
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cublas_workspace_config",
        "torch_use_deterministic_algorithms",
    }
    assert set(cfg.keys()) == expected_keys
    assert isinstance(cfg["cudnn_deterministic"], bool)
    assert isinstance(cfg["cudnn_benchmark"], bool)
    assert isinstance(cfg["torch_use_deterministic_algorithms"], bool)
    # cublas_workspace_config is the env var value or None if unset
    assert cfg["cublas_workspace_config"] is None or isinstance(cfg["cublas_workspace_config"], str)


def test_torch_cpu_metadata_has_no_cuda_driver_version(torch_metadata):
    """cuda_driver_version is only included when device=cuda and nvidia-smi is
    available. CPU-device bakes (or hosts without NVIDIA drivers) should omit it."""
    # On CPU device, the field is unconditionally absent.
    assert torch_metadata["device"] == "cpu"
    assert "cuda_driver_version" not in torch_metadata


# ----------------- validation back-compat -----------------


def test_validate_metadata_accepts_v3_with_provenance(tmp_path: Path):
    """A 3.0 single-split metadata with the new fields passes validate_metadata."""
    metadata = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {
            "name": "whestbench_explicit_per_mlp_seeds",
            "version": "3.0",
        },
        "n_mlps": 4,
        "n_samples": 1_000_000_000,
        "width": 256,
        "depth": 8,
        "whestbench_version": "0.3.0",
        "flopscope_version": "0.3.x",
        "bake_config": {
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
            "torch_use_deterministic_algorithms": False,
        },
    }
    validate_metadata(metadata, allow_partial=False)  # no raise


def test_validate_metadata_accepts_v3_without_provenance(tmp_path: Path):
    """Datasets baked BEFORE this PR don't have the new fields. They must
    continue to validate."""
    metadata = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "flopscope",
        "seed_protocol": {
            "name": "whestbench_explicit_per_mlp_seeds",
            "version": "3.0",
        },
        "n_mlps": 4,
        "n_samples": 1_000_000_000,
        "width": 256,
        "depth": 8,
    }
    validate_metadata(metadata, allow_partial=False)  # no raise


# ----------------- helper module -----------------


def test_provenance_helpers_degrade_gracefully():
    """The provenance helpers must never raise — return "unknown"/None on failure."""
    from whestbench._provenance import (
        flopscope_version,
        nvidia_driver_version,
        whestbench_version,
    )

    # Whestbench + flopscope are installed via the test env; should return real strings.
    assert isinstance(whestbench_version(), str)
    assert isinstance(flopscope_version(), str)

    # nvidia-smi may or may not be present; either is acceptable.
    drv = nvidia_driver_version()
    assert drv is None or isinstance(drv, str)
