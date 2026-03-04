"""Shared hardware fingerprinting for dataset staleness and CLI reporting."""

from __future__ import annotations

import os
import platform
import socket
from typing import Any

import numpy as np

try:
    import psutil  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover
    psutil = None


def collect_hardware_fingerprint() -> dict[str, Any]:
    """Collect a hardware fingerprint dict for the current machine.

    Returns a dict containing hostname, OS info, CPU details,
    RAM statistics, and numpy version. Fields that require ``psutil``
    fall back to ``None`` when the library is unavailable.
    """
    fp: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_brand": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count() or 1,
        "cpu_count_physical": None,
        "ram_total_bytes": None,
        "ram_available_bytes": None,
        "numpy_version": np.__version__,
    }
    if psutil is not None:
        try:
            fp["cpu_count_physical"] = psutil.cpu_count(logical=False)
        except Exception:
            pass
        try:
            mem = psutil.virtual_memory()
            fp["ram_total_bytes"] = int(mem.total)
            fp["ram_available_bytes"] = int(mem.available)
        except Exception:
            pass
    return fp


_STALENESS_KEYS = ("machine", "cpu_brand", "cpu_count_logical", "ram_total_bytes")


def hardware_matches(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Return True if the stored and current fingerprints match on key fields.

    Compares machine architecture, CPU brand, logical core count, and
    total RAM — the fields most likely to affect sampling baseline times.
    """
    return all(stored.get(k) == current.get(k) for k in _STALENESS_KEYS)
