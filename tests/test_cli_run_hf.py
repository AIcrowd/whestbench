"""Tests for `whest run --dataset` URL resolution."""

from __future__ import annotations

import pytest

from whestbench.cli import _resolve_dataset_arg


def test_resolves_local_relative_path():
    repo, rev, is_local = _resolve_dataset_arg("./eval-v1", revision=None)
    assert is_local is True
    assert repo == "./eval-v1"


def test_resolves_local_absolute_path():
    repo, rev, is_local = _resolve_dataset_arg("/tmp/x", revision=None)
    assert is_local is True


def test_resolves_hf_url_scheme():
    repo, rev, is_local = _resolve_dataset_arg(
        "hf://aicrowd/arc-whestbench-2026-eval@v1", revision=None
    )
    assert is_local is False
    assert repo == "aicrowd/arc-whestbench-2026-eval"
    assert rev == "v1"


def test_resolves_hf_url_without_tag():
    repo, rev, is_local = _resolve_dataset_arg(
        "hf://aicrowd/arc-whestbench-2026-eval", revision=None
    )
    assert is_local is False
    assert repo == "aicrowd/arc-whestbench-2026-eval"
    assert rev is None


def test_resolves_repo_with_revision_flag():
    repo, rev, is_local = _resolve_dataset_arg("aicrowd/arc-whestbench-2026-eval", revision="v1")
    assert is_local is False
    assert repo == "aicrowd/arc-whestbench-2026-eval"
    assert rev == "v1"


def test_rejects_bare_repo_without_revision_or_prefix():
    with pytest.raises(SystemExit, match="hf://"):
        _resolve_dataset_arg("aicrowd/arc-whestbench-2026-eval", revision=None)
