"""HuggingFace-specific progress glue: preflight + Rich tqdm bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from huggingface_hub import HfApi, try_to_load_from_cache


@dataclass(frozen=True)
class HFPreflight:
    """Result of probing HF for download size + cache state for a (repo, revision).

    Returned by ``hf_preflight``. ``None`` is used at the call-site to mean
    "preflight failed; proceed without info" — never an HFPreflight with all
    zeros.
    """

    repo_id: str
    revision: "str | None"
    file_count: int
    total_bytes: int
    is_cached: bool
    files: List[Tuple[str, int]]  # (rfilename, size_bytes)


def hf_preflight(
    repo_id: str,
    *,
    revision: Optional[str],
    token: Optional[str] = None,
    split: Optional[str] = None,
) -> "Optional[HFPreflight]":
    """Probe HF for download size + cache state before a load/download.

    Returns ``None`` on any HF error (network, 401, gated repo, etc.) — callers
    treat this as "proceed without info" and the progress bar adapts.

    For datasets:
    - if ``split`` is given, only ``data/<split>-*.parquet`` is counted as
      data; ``metadata.json`` + ``README.md`` are always included.
    - if ``split`` is None, all ``data/*.parquet`` files count.
    """
    try:
        api = HfApi(token=token)
        info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    except Exception:  # noqa: BLE001 — preflight is best-effort; any failure → None
        return None

    relevant: list[tuple[str, int]] = []
    for sib in info.siblings:
        name = sib.rfilename
        size = int(getattr(sib, "size", 0) or 0)
        keep = False
        if name in ("metadata.json", "README.md"):
            keep = True
        elif name.startswith("data/") and name.endswith(".parquet"):
            if split is None:
                keep = True
            else:
                base = name[len("data/") :]
                keep = base.startswith(f"{split}-")
        if keep:
            relevant.append((name, size))

    if not relevant:
        return None

    # Cache probe: try_to_load_from_cache returns a path string, None, or
    # _CACHED_NO_EXIST. We treat anything other than a real str path as "not
    # cached" so a missing file or the sentinel both flip is_cached to False.
    resolved_revision = info.sha if revision is None else revision
    all_cached = True
    for name, _size in relevant:
        cached = try_to_load_from_cache(
            repo_id=repo_id,
            filename=name,
            repo_type="dataset",
            revision=resolved_revision,
        )
        if cached is None or not isinstance(cached, str):
            all_cached = False
            break

    return HFPreflight(
        repo_id=repo_id,
        revision=revision,
        file_count=len(relevant),
        total_bytes=sum(s for _, s in relevant),
        is_cached=all_cached,
        files=relevant,
    )
