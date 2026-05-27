"""HuggingFace-specific progress glue: preflight + Rich tqdm bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import tqdm as _tqdm_mod
from huggingface_hub import HfApi, try_to_load_from_cache

# ---------------------------------------------------------------------------
# Module-level mutable state
#
# ``_ACTIVE_RICH_PROGRESS`` is the only mutable global in this module. It is
# set to a Rich ``Progress`` instance by ``hf_download`` / ``hf_upload`` for
# the lifetime of their context-manager body, and reset to ``None`` in their
# ``finally`` blocks. When set, ``RichHFTqdm`` instances created by HF Hub's
# internal download/upload code path (via the monkey-patched
# ``huggingface_hub.utils.tqdm``) forward their progress events into the
# Progress; when ``None`` they fall back to a no-op so that callers outside a
# ``hf_download``/``hf_upload`` context do not crash.
_ACTIVE_RICH_PROGRESS: Any = None


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


class RichHFTqdm(_tqdm_mod.tqdm):
    """tqdm subclass that mirrors progress into our active Rich Progress.

    HF Hub creates one of these whenever it would normally create a ``tqdm.tqdm``
    bar (per-file downloads, etc.). If a Rich ``Progress`` is currently active
    (i.e. ``_ACTIVE_RICH_PROGRESS is not None``), every ``__init__`` / ``update``
    / ``close`` forwards into it via ``add_task`` / ``update`` / ``remove_task``.
    When no Progress is active the subclass behaves like a vanilla tqdm.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        prog = _ACTIVE_RICH_PROGRESS
        if prog is not None:
            self._rich_task_id: "int | None" = prog.add_task(
                self.desc or "Downloading", total=self.total
            )
        else:
            self._rich_task_id = None

    def update(self, n: int = 1) -> None:  # type: ignore[override]
        super().update(n)
        prog = _ACTIVE_RICH_PROGRESS
        if prog is not None and self._rich_task_id is not None:
            prog.update(self._rich_task_id, completed=self.n, total=self.total)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        prog = _ACTIVE_RICH_PROGRESS
        if prog is not None and self._rich_task_id is not None:
            try:
                prog.remove_task(self._rich_task_id)
            finally:
                self._rich_task_id = None
