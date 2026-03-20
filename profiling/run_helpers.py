"""Shared helpers for the profiling orchestrator and collector."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _git_short_hash() -> str:
    """Return first 7 chars of HEAD commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _git_full_hash() -> str:
    """Return full HEAD commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    return bool(result.stdout.strip())


def generate_run_id(override: Optional[str] = None) -> str:
    """Generate a run ID in the format YYYY-MM-DD-HHMMSS-<git-hash>[-dirty].

    Args:
        override: If provided, return this value directly.

    Returns:
        A unique run identifier string.
    """
    if override:
        return override
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    git_hash = _git_short_hash()
    run_id = f"{now}-{git_hash}"
    if _git_is_dirty():
        run_id += "-dirty"
    return run_id


def git_metadata() -> Dict[str, Any]:
    """Return git metadata for embedding in run results."""
    return {
        "commit": _git_full_hash(),
        "commit_short": _git_short_hash(),
        "dirty": _git_is_dirty(),
    }


def load_infra_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load .infra-config.json created by setup_infra.sh.

    Args:
        config_path: Override path. Defaults to profiling/.infra-config.json.

    Returns:
        Dict with keys: region, account_id, s3_bucket, ecr_repo_uri,
        cluster_name, cluster_arn, execution_role_arn, task_role_arn,
        log_group, image_uri.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent / ".infra-config.json")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Run setup_infra.sh first."
        )
    with open(path) as f:
        return json.load(f)
