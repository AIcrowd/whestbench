"""Tests for whestbench.hub.publish_dataset (mocked HF API)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def _bake_small(tmp_path: Path) -> Path:
    from whestbench.dataset import create_dataset

    out = tmp_path / "ds"
    create_dataset(n_mlps=2, n_samples=50, width=4, depth=2, seed=1, output_path=out)
    return out


@patch("whestbench.hub.HfApi")
@patch("whestbench.hub.create_repo")
def test_publish_uploads_and_returns_sha(mock_create_repo, mock_hfapi, tmp_path: Path):
    from whestbench.hub import publish_dataset

    out = _bake_small(tmp_path)
    api_instance = MagicMock()
    mock_hfapi.return_value = api_instance
    api_instance.upload_folder.return_value.oid = "abc123def456"

    sha = publish_dataset(out, repo_id="aicrowd/test-ds", token="t")
    assert sha == "abc123def456"
    mock_create_repo.assert_called_once()
    api_instance.upload_folder.assert_called_once()
    api_instance.create_tag.assert_not_called()


@patch("whestbench.hub.HfApi")
@patch("whestbench.hub.create_repo")
def test_publish_creates_tag_when_specified(mock_create_repo, mock_hfapi, tmp_path: Path):
    from whestbench.hub import publish_dataset

    out = _bake_small(tmp_path)
    api_instance = MagicMock()
    mock_hfapi.return_value = api_instance
    api_instance.upload_folder.return_value.oid = "deadbeef"

    publish_dataset(out, repo_id="aicrowd/test-ds", tag="v1", token="t")
    api_instance.create_tag.assert_called_once()
    call_kwargs = api_instance.create_tag.call_args.kwargs
    assert call_kwargs["tag"] == "v1"
    assert call_kwargs["revision"] == "deadbeef"


@patch("whestbench.hub.HfApi")
@patch("whestbench.hub.create_repo")
def test_publish_rerenders_readme_with_repo_id(mock_create_repo, mock_hfapi, tmp_path: Path):
    from whestbench.hub import publish_dataset

    out = _bake_small(tmp_path)
    api_instance = MagicMock()
    mock_hfapi.return_value = api_instance
    api_instance.upload_folder.return_value.oid = "x"

    publish_dataset(out, repo_id="aicrowd/test-ds", tag="v1", token="t")

    readme = (out / "README.md").read_text()
    assert "aicrowd/test-ds" in readme
    assert "<your-repo>" not in readme
    assert 'revision="v1"' in readme


@patch("whestbench.hub.HfApi")
@patch("whestbench.hub.create_repo")
def test_publish_supports_private(mock_create_repo, mock_hfapi, tmp_path: Path):
    from whestbench.hub import publish_dataset

    out = _bake_small(tmp_path)
    api_instance = MagicMock()
    mock_hfapi.return_value = api_instance
    api_instance.upload_folder.return_value.oid = "y"

    publish_dataset(out, repo_id="aicrowd/private-ds", token="t", private=True)
    create_kwargs = mock_create_repo.call_args.kwargs
    assert create_kwargs["private"] is True
