"""Shared UX helpers: formatters, message styles, and progress context managers.

All helpers in this module respect:
- ``HF_HUB_DISABLE_PROGRESS_BARS=1`` (silences progress)
- ``NO_COLOR=1`` (Rich auto-respects)
- ``--json`` and ``no_rich`` modes (toggled by callers via ``quiet=`` flag)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional, Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


class ProgressHandle(Protocol):
    """Public surface for a progress bar handle.

    Both the real Rich-backed handle and the disabled no-op handle satisfy
    this protocol, so callers can write code that runs identically in either
    mode.
    """

    def advance(self, n: int) -> None: ...

    def update(self, *, completed: int) -> None: ...


_DEFAULT_CONSOLE: Optional[Console] = None


def _get_console(console: Optional[Console]) -> Console:
    """Return the provided console, or the lazy module-level default one.

    The default ``Console`` is constructed on first use and cached for the
    process lifetime. Rich's ``Console`` is internally thread-safe, so no
    locking is required around the lazy init.
    """
    if console is not None:
        return console
    global _DEFAULT_CONSOLE
    if _DEFAULT_CONSOLE is None:
        _DEFAULT_CONSOLE = Console()
    return _DEFAULT_CONSOLE


_HF_TRUTHY_VALUES = frozenset({"1", "on", "true", "yes"})


def _progress_disabled(quiet: bool) -> bool:
    """Return True when progress bars should be suppressed.

    Suppressed if ``quiet`` is True, or if ``HF_HUB_DISABLE_PROGRESS_BARS`` is
    set to a truthy value. This follows HF Hub's own env-var convention: only
    ``{"1", "ON", "TRUE", "YES"}`` (case-insensitive) count as truthy. Any
    other value — including ``""``, ``"0"``, ``"false"``, ``"no"``, ``"off"``,
    or unrecognised strings — leaves progress enabled.
    """
    if quiet:
        return True
    raw = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    if raw is None:
        return False
    return raw.strip().lower() in _HF_TRUTHY_VALUES


class _NullHandle(ProgressHandle):
    """No-op progress handle returned when bars are disabled.

    Inherits from :class:`ProgressHandle` nominally so that drift in the
    Protocol surface is caught at type-check time.
    """

    def advance(self, n: int) -> None:  # noqa: ARG002 - signature matches real handle
        return None

    def update(self, *, completed: int) -> None:  # noqa: ARG002
        return None


class _RealProgressHandle(ProgressHandle):
    """Thin wrapper around a Rich ``Progress`` task with the public API.

    Exposes ``advance(n)`` and ``update(*, completed=)`` — the only two
    operations callers need for the bytes/count bars. Inherits from
    :class:`ProgressHandle` nominally so that drift in the Protocol surface
    is caught at type-check time.
    """

    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self, n: int) -> None:
        self._progress.update(self._task_id, advance=n)

    def update(self, *, completed: int) -> None:
        self._progress.update(self._task_id, completed=completed)


def format_bytes(n_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Uses 1024-based units (KB = 1024 B, etc.) with one decimal place from KB
    upward. Below 1024 bytes returns plain bytes. Values just under the next
    unit (e.g. ``1024**2 - 1``) roll over to the next unit rather than
    rendering as ``"1024.0 KB"``.
    """
    if n_bytes < 0:
        raise ValueError(f"format_bytes requires a non-negative count, got {n_bytes!r}")
    if n_bytes < 1024:
        return f"{n_bytes} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    size = float(n_bytes)
    for unit in units:
        size /= 1024.0
        # If we're at the last unit, render as-is — there's nothing to roll
        # over to.
        if unit == units[-1]:
            return f"{size:.1f} {unit}"
        # Use this unit only if size will not render as "1024.0" after .1f
        # rounding. Otherwise continue to the next unit so values just shy of
        # the next boundary (e.g. 1024**2 - 1) don't render as "1024.0 KB".
        if round(size, 1) < 1024.0:
            return f"{size:.1f} {unit}"
    raise AssertionError("unreachable")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string.

    - <1s: "Nms" (integer milliseconds)
    - <60s: "X.Ys" (one decimal)
    - <1h:  "Xm Ys"
    - >=1h: "Xh Ym Zs"
    """
    if seconds < 0:
        raise ValueError(f"format_duration requires non-negative seconds, got {seconds!r}")
    if seconds < 1.0:
        return f"{int(round(seconds * 1000))}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


def format_throughput(n_bytes: int, seconds: float) -> str:
    """Bytes per second, formatted via ``format_bytes``. Zero/negative time → ``— /s``."""
    if seconds <= 0:
        return "— /s"
    bps = int(n_bytes / seconds)
    return f"{format_bytes(bps)}/s"


class _Say:
    """Styled message emitter used as the module-level ``say`` singleton.

    Each method takes ``(msg, *, console=None, quiet=False)``. ``quiet=True``
    makes the call a no-op; callers pass ``quiet=True`` for ``--json`` /
    ``no_rich`` modes. The configured Rich ``Console`` already respects
    ``NO_COLOR``.
    """

    def intent(self, msg: str, *, console: Optional[Console] = None, quiet: bool = False) -> None:
        """Announce an upcoming long-running action (one per verb)."""
        if quiet:
            return
        _get_console(console).print(f"[bold]{msg}[/bold]")

    def step(self, msg: str, *, console: Optional[Console] = None, quiet: bool = False) -> None:
        """Note an in-progress sub-step (informational, dimmed)."""
        if quiet:
            return
        _get_console(console).print(f"[dim]{msg}[/dim]")

    def ok(self, msg: str, *, console: Optional[Console] = None, quiet: bool = False) -> None:
        """Report a successful outcome with a leading check mark."""
        if quiet:
            return
        _get_console(console).print(f"[bold green]✓[/bold green] {msg}")

    def warn(self, msg: str, *, console: Optional[Console] = None, quiet: bool = False) -> None:
        """Flag a non-default behaviour with a leading warning glyph."""
        if quiet:
            return
        _get_console(console).print(f"[bold yellow]⚠[/bold yellow] {msg}")

    def hint(self, msg: str, *, console: Optional[Console] = None, quiet: bool = False) -> None:
        """Offer an actionable tip — never a warning, never repeated noisily."""
        if quiet:
            return
        _get_console(console).print(f"[dim]tip: {msg}[/dim]")


say = _Say()


@contextmanager
def progress_bytes(
    *,
    total: int,
    label: str,
    console: Optional[Console] = None,
    quiet: bool = False,
) -> Iterator[ProgressHandle]:
    """Yield a handle for a byte-denominated Rich progress bar.

    The handle exposes ``advance(n)`` and ``update(*, completed=)``. The bar is
    suppressed when ``quiet`` is True or when ``HF_HUB_DISABLE_PROGRESS_BARS``
    is set to a truthy value — in that case, a no-op handle is yielded.
    """
    if _progress_disabled(quiet):
        yield _NullHandle()
        return
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=_get_console(console),
        transient=False,
    )
    with progress:
        task_id = progress.add_task(label, total=total)
        yield _RealProgressHandle(progress, task_id)


@contextmanager
def status(
    label: str,
    *,
    console: Optional[Console] = None,
    quiet: bool = False,
) -> Iterator[None]:
    """Yield while showing a Rich spinner with ``label``.

    Useful for unknown-duration steps (e.g. "Loading from cache",
    "Resolving revision"). No bar is shown when ``quiet`` is True or when
    ``HF_HUB_DISABLE_PROGRESS_BARS`` is set to a truthy value.
    """
    if _progress_disabled(quiet):
        yield
        return
    with _get_console(console).status(label):
        yield


@contextmanager
def progress_count(
    *,
    total: int,
    label: str,
    console: Optional[Console] = None,
    quiet: bool = False,
) -> Iterator[ProgressHandle]:
    """Yield a handle for a count-denominated Rich progress bar.

    Same shape as :func:`progress_bytes` but with M-of-N + elapsed/remaining
    columns instead of bytes/throughput — used for per-MLP sampling, scoring,
    and similar count-style loops.
    """
    if _progress_disabled(quiet):
        yield _NullHandle()
        return
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_get_console(console),
        transient=False,
    )
    with progress:
        task_id = progress.add_task(label, total=total)
        yield _RealProgressHandle(progress, task_id)
