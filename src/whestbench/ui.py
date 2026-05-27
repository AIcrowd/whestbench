"""Shared UX helpers: formatters, message styles, and progress context managers.

All helpers in this module respect:
- ``HF_HUB_DISABLE_PROGRESS_BARS=1`` (silences progress)
- ``NO_COLOR=1`` (Rich auto-respects)
- ``--json`` and ``no_rich`` modes (toggled by callers via ``quiet=`` flag)
"""

from __future__ import annotations


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
