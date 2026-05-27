"""HuggingFace-specific progress glue: preflight + Rich tqdm bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


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
