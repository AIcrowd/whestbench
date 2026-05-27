"""HuggingFace-specific progress glue: preflight + Rich tqdm bridge."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Literal, Optional, Tuple, Union

import huggingface_hub.utils
from huggingface_hub import HfApi, try_to_load_from_cache
from huggingface_hub.utils.tqdm import (
    tqdm as _hf_tqdm,  # pyright: ignore[reportPrivateImportUsage]
)
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

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
    for sib in info.siblings or ():
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


class RichHFTqdm(_hf_tqdm):
    """tqdm subclass that mirrors progress into our active Rich Progress.

    Subclasses HF Hub's own ``tqdm`` (not vanilla ``tqdm.std.tqdm``) so we keep
    HF-specific behaviour like the ``name=`` kwarg used by group-disable
    (``huggingface_hub.utils.tqdm`` strips ``name`` before delegating, then sets
    ``disable=True`` if that group is muted via ``HF_HUB_DISABLE_PROGRESS_BARS``
    or an explicit ``disable_progress_bars(name)``).

    When a Rich ``Progress`` is currently active (``_ACTIVE_RICH_PROGRESS is not
    None``) and the bar is NOT disabled, every ``__init__`` / ``update`` /
    ``close`` forwards into it via ``add_task`` / ``update`` / ``remove_task``.
    Otherwise the subclass behaves like a vanilla tqdm — which includes the
    disabled case, where ``tqdm.std.tqdm`` deliberately does not set ``desc`` /
    other display attrs, so we must NOT try to read them.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        prog = _ACTIVE_RICH_PROGRESS
        # When ``disable=True`` (set either explicitly or by HF Hub's group-disable
        # path when ``HF_HUB_DISABLE_PROGRESS_BARS`` is on), tqdm skips setting
        # ``self.desc`` / display attrs entirely. Reading them would AttributeError,
        # so we short-circuit Rich registration in that case.
        if prog is None or getattr(self, "disable", False):
            self._rich_task_id: "int | None" = None
            return
        self._rich_task_id = prog.add_task(self.desc or "Downloading", total=self.total)

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


DownloadMode = Literal["cache_hit", "materialize", "streaming"]


@contextmanager
def hf_download(
    console: Console,
    *,
    title: str,
    preflight: "Optional[HFPreflight]",  # noqa: ARG001 — reserved for future label tweaks
    mode: DownloadMode,
    quiet: bool = False,
) -> Iterator[None]:
    """Wrap an HF download call with mode-appropriate progress UI.

    Modes:
        cache_hit   — Rich Status spinner only. Does NOT monkey-patch tqdm.
        materialize — Full bytes/speed/ETA Progress bar; monkey-patches
                      ``huggingface_hub.utils.tqdm`` to ``RichHFTqdm`` so HF's
                      internal progress events route into the bar.
        streaming   — Lighter Progress (spinner + speed, no total/bar);
                      same monkey-patch.

    The patch is restored in a ``try/finally`` that survives exceptions raised
    inside the context body.
    """
    global _ACTIVE_RICH_PROGRESS

    if quiet:
        yield
        return

    if mode == "cache_hit":
        # Cache-hit path: a short spinner is all we want. Crucially this branch
        # does NOT touch ``_ACTIVE_RICH_PROGRESS`` or ``huggingface_hub.utils.tqdm``,
        # so nested HF calls fall back to vanilla tqdm.
        with console.status(f"Loading {title} from cache…"):
            yield
        return

    # materialize or streaming
    if mode == "materialize":
        columns: tuple[Any, ...] = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )
    else:  # streaming — no total known up-front, so drop Bar/Download/ETA columns.
        columns = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TransferSpeedColumn(),
        )

    progress = Progress(*columns, console=console, transient=False)
    # ``huggingface_hub.utils.tqdm`` is the live class HF Hub instantiates for
    # per-file progress bars; we swap it for ``RichHFTqdm`` for the duration of
    # this context. Pyright flags it as a private import because it's not in
    # ``__all__``, but it's documented HF internals (see hf_hub.utils.tqdm).
    original_tqdm = huggingface_hub.utils.tqdm  # pyright: ignore[reportPrivateImportUsage]

    with progress:
        _ACTIVE_RICH_PROGRESS = progress
        huggingface_hub.utils.tqdm = RichHFTqdm  # pyright: ignore[reportPrivateImportUsage]
        try:
            yield
        finally:
            huggingface_hub.utils.tqdm = original_tqdm  # pyright: ignore[reportPrivateImportUsage]
            _ACTIVE_RICH_PROGRESS = None


def _du_local(p: Path) -> int:
    """Sum sizes of all regular files under ``p`` (recursive, symlink-safe).

    Each ``stat()`` call is wrapped in ``try/except OSError`` so a vanished or
    unreadable file silently contributes zero rather than aborting the whole
    walk — preflight-style "best effort" sizing.
    """
    total = 0
    for f in p.rglob("*"):
        if f.is_file() and not f.is_symlink():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


@contextmanager
def hf_upload(
    console: Console,
    *,
    title: str,
    local_dir: Union[Path, str],
    quiet: bool = False,
) -> Iterator[None]:
    """Wrap an HF upload call with a bytes Progress bar.

    Computes total bytes from ``local_dir`` once at entry; monkey-patches
    ``huggingface_hub.utils.tqdm`` for the duration so per-file events route
    into the bar. The patch is restored in a ``try/finally`` that survives
    exceptions raised inside the context body.

    When ``quiet=True`` the call is a passthrough — no bar, no tqdm swap.
    """
    global _ACTIVE_RICH_PROGRESS

    if quiet:
        yield
        return

    local_dir = Path(local_dir)
    total = _du_local(local_dir)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    original_tqdm = huggingface_hub.utils.tqdm  # pyright: ignore[reportPrivateImportUsage]

    with progress:
        _ACTIVE_RICH_PROGRESS = progress
        huggingface_hub.utils.tqdm = RichHFTqdm  # pyright: ignore[reportPrivateImportUsage]
        # Seed a single overall task representing the whole upload — HF Hub
        # will spawn its own per-file RichHFTqdm bars on top.
        progress.add_task(f"Uploading → {title}", total=total)
        try:
            yield
        finally:
            huggingface_hub.utils.tqdm = original_tqdm  # pyright: ignore[reportPrivateImportUsage]
            _ACTIVE_RICH_PROGRESS = None
