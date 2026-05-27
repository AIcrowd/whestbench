"""Shared UX helpers: formatters, message styles, and progress context managers.

All helpers in this module respect:
- ``HF_HUB_DISABLE_PROGRESS_BARS=1`` (silences progress)
- ``NO_COLOR=1`` (Rich auto-respects)
- ``--json`` and ``no_rich`` modes (toggled by callers via ``quiet=`` flag)
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console


def _get_console(console: Optional[Console]) -> Console:
    """Return the provided console, or a fresh default one."""
    if console is not None:
        return console
    return Console()


def format_bytes(n_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Uses 1024-based units (KB = 1024 B, etc.) with one decimal place from KB
    upward. Below 1024 bytes returns plain bytes.
    """
    if n_bytes < 0:
        raise ValueError(f"format_bytes requires a non-negative count, got {n_bytes!r}")
    if n_bytes < 1024:
        return f"{n_bytes} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    size = float(n_bytes)
    for unit in units:
        size /= 1024.0
        if size < 1024.0 or unit == units[-1]:
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
