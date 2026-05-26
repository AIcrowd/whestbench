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


# -----------------------------------------------------------------------------
# metadata_file_hash dispatch (local vs hf://)
# -----------------------------------------------------------------------------


def test_metadata_file_hash_local_directory(tmp_path: Path):
    """Local directory path → reads <path>/metadata.json from disk."""
    import json as _json

    from whestbench.dataset_io import metadata_file_hash

    md = {"schema_version": "3.0", "n_mlps": 1}
    (tmp_path / "metadata.json").write_text(_json.dumps(md))
    h = metadata_file_hash(tmp_path)
    assert isinstance(h, str)
    assert len(h) == 64  # sha256 hex


def test_metadata_file_hash_hf_url_uses_huggingface_hub(monkeypatch, tmp_path: Path):
    """hf://owner/repo[@rev] URL → downloads via huggingface_hub.hf_hub_download.

    Monkeypatch hf_hub_download to return a local file so the test doesn't
    require network. Verify the function is called with the expected kwargs.
    """
    import json as _json

    md_path = tmp_path / "metadata.json"
    md_path.write_text(_json.dumps({"schema_version": "3.0", "n_mlps": 7}))

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(md_path)

    import huggingface_hub
    from huggingface_hub import hf_hub_download  # noqa: F401 (ensure module loaded)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    from whestbench.dataset_io import metadata_file_hash

    h = metadata_file_hash("hf://aicrowd/some-repo@v1-test")
    assert len(h) == 64
    assert calls == [
        {
            "repo_id": "aicrowd/some-repo",
            "filename": "metadata.json",
            "repo_type": "dataset",
            "revision": "v1-test",
        }
    ]


def test_metadata_file_hash_hf_url_no_embedded_revision(monkeypatch, tmp_path: Path):
    """hf://owner/repo with no @<rev> → revision=None passed to hf_hub_download."""
    import json as _json

    md_path = tmp_path / "metadata.json"
    md_path.write_text(_json.dumps({"schema_version": "3.0"}))

    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(md_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    from whestbench.dataset_io import metadata_file_hash

    metadata_file_hash("hf://aicrowd/some-repo")
    assert captured["revision"] is None
    assert captured["repo_id"] == "aicrowd/some-repo"


def test_metadata_file_hash_kwarg_revision_overrides_when_no_embedded(monkeypatch, tmp_path: Path):
    """Bare hf:// URL + explicit revision= kwarg → kwarg wins."""
    import json as _json

    md_path = tmp_path / "metadata.json"
    md_path.write_text(_json.dumps({"schema_version": "3.0"}))

    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(md_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    from whestbench.dataset_io import metadata_file_hash

    metadata_file_hash("hf://aicrowd/some-repo", revision="v2")
    assert captured["revision"] == "v2"


def test_metadata_file_hash_local_and_hf_agree_on_same_content(monkeypatch, tmp_path: Path):
    """Local and hf:// paths return the same hash when the bytes match."""
    import json as _json

    md = {"schema_version": "3.0", "n_mlps": 42}
    (tmp_path / "metadata.json").write_text(_json.dumps(md))

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **kw: str(tmp_path / "metadata.json")
    )

    from whestbench.dataset_io import metadata_file_hash

    h_local = metadata_file_hash(tmp_path)
    h_hf = metadata_file_hash("hf://aicrowd/whatever@v1")
    assert h_local == h_hf
