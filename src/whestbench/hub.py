"""HuggingFace Hub publishing for whestbench datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo

from .dataset_io import METADATA_FILE, README_FILE, generate_readme


def _rerender_readme_with_repo(local_dir: Path, *, repo_id: str, revision: str) -> None:
    """Re-render README.md with concrete repo_id and revision before upload."""
    md = json.loads((local_dir / METADATA_FILE).read_text())
    parquet_files = list((local_dir / "data").glob("*.parquet"))
    if len(parquet_files) != 1:
        return
    split = parquet_files[0].name.split("-")[0]
    ds_size = md.get("n_mlps", 0)
    (local_dir / README_FILE).write_text(
        generate_readme(md, split=split, ds_size=ds_size, repo_id=repo_id, revision=revision)
    )


def publish_dataset(
    local_dir: "Path | str",
    *,
    repo_id: str,
    tag: Optional[str] = None,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
    repo_exist_ok: bool = True,
) -> str:
    """Upload a baked dataset directory to HF Hub.

    Re-renders README.md with the actual repo_id and revision (tag) before
    upload so the published card has real values, not placeholders.

    Args:
        local_dir: Directory containing data/, metadata.json, README.md.
        repo_id: e.g. "aicrowd/arc-whestbench-2026".
        tag: If provided, creates a git tag pointing at the new commit.
        token: HF Hub auth token; falls back to HF auth cache.
        commit_message: Commit message; default is auto-generated.
        private: If creating the repo, mark it private.
        repo_exist_ok: If True, don't error when the repo already exists.

    Returns:
        The commit SHA from the upload.
    """
    local_dir = Path(local_dir)
    revision_label = tag or "main"
    _rerender_readme_with_repo(local_dir, repo_id=repo_id, revision=revision_label)

    create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=repo_exist_ok,
        token=token,
    )

    api = HfApi(token=token)
    commit_info = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or "Upload whestbench dataset",
        token=token,
    )

    if tag is not None:
        api.create_tag(
            repo_id=repo_id,
            tag=tag,
            revision=commit_info.oid,
            repo_type="dataset",
            token=token,
        )

    return commit_info.oid
