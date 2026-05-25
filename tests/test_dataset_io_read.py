"""Tests for dataset_io.read_metadata and validate_metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whestbench.dataset_io import (
    SCHEMA_FORMAT,
    SCHEMA_VERSION,
    SEED_PROTOCOL_VERSION,
    InvalidDatasetError,
    read_metadata,
    validate_metadata,
)


def _good_metadata(**overrides):
    base = {
        "schema_version": SCHEMA_VERSION,
        "format": SCHEMA_FORMAT,
        "backend": "flopscope",
        "seed_protocol": {
            "name": "whestbench_seedsequence_hierarchy",
            "version": SEED_PROTOCOL_VERSION,
            "seeded": True,
        },
        "created_at_utc": "2026-05-25T00:00:00+00:00",
        "seed": 42,
        "n_mlps": 1,
        "n_samples": 100,
        "width": 4,
        "depth": 2,
        "hardware": {},
    }
    base.update(overrides)
    return base


def test_read_metadata_round_trip(tmp_path: Path):
    out = tmp_path / "ds"
    out.mkdir()
    (out / "metadata.json").write_text(json.dumps(_good_metadata()))
    assert read_metadata(out) == _good_metadata()


def test_read_metadata_raises_if_file_missing(tmp_path: Path):
    out = tmp_path / "ds"
    out.mkdir()
    with pytest.raises(InvalidDatasetError, match="metadata.json"):
        read_metadata(out)


def test_validate_accepts_good_metadata():
    validate_metadata(_good_metadata())  # no raise


def test_validate_rejects_missing_schema_version():
    bad = _good_metadata()
    del bad["schema_version"]
    with pytest.raises(InvalidDatasetError, match="schema_version"):
        validate_metadata(bad)


def test_validate_rejects_wrong_schema_version():
    bad = _good_metadata(schema_version="2.4")
    with pytest.raises(InvalidDatasetError, match=r"schema_version.*3\.0"):
        validate_metadata(bad)


def test_validate_rejects_wrong_seed_protocol_version():
    bad = _good_metadata(seed_protocol={"name": "x", "version": "1.0", "seeded": True})
    with pytest.raises(InvalidDatasetError, match="seed_protocol"):
        validate_metadata(bad)


def test_validate_rejects_partial_metadata_by_default():
    bad = _good_metadata(is_partial=True, mlp_range=[0, 5], total_n_mlps=10)
    with pytest.raises(InvalidDatasetError, match="partial"):
        validate_metadata(bad)


def test_validate_accepts_partial_when_allowed():
    partial = _good_metadata(is_partial=True, mlp_range=[0, 5], total_n_mlps=10)
    validate_metadata(partial, allow_partial=True)  # no raise
