"""Shared hardware fingerprinting for dataset staleness and CLI reporting."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Any

import numpy as np

try:
    import psutil  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover
    psutil = None


def _physical_core_count_fallback() -> int | None:
    """Try OS-native methods to get physical core count without psutil."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True, timeout=5,
            )
            return int(out.strip())
        if system == "Linux":
            out = subprocess.check_output(
                ["nproc", "--all"], text=True, timeout=5,
            )
            # nproc --all gives logical; parse /sys for physical
            try:
                cores = set()
                with open("/sys/devices/system/cpu/online") as f:
                    # e.g. "0-15"
                    pass
                import glob

                for path in glob.glob(
                    "/sys/devices/system/cpu/cpu[0-9]*/topology/core_id"
                ):
                    with open(path) as f:
                        cores.add(f.read().strip())
                if cores:
                    return len(cores)
            except Exception:
                pass
            return int(out.strip())
    except Exception:
        pass
    return None


def _ram_total_fallback() -> int | None:
    """Try OS-native methods to get total RAM without psutil."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5,
            )
            return int(out.strip())
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Value is in kB
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def collect_hardware_fingerprint() -> dict[str, Any]:
    """Collect a hardware fingerprint dict for the current machine.

    Returns a dict containing hostname, OS info, CPU details,
    RAM statistics, and numpy version. Uses ``psutil`` when available,
    with OS-native fallbacks (sysctl on macOS, /proc on Linux) to
    ensure fields are populated on all major platforms.
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

    # OS-native fallbacks when psutil didn't provide values
    if fp["cpu_count_physical"] is None:
        fp["cpu_count_physical"] = _physical_core_count_fallback()
    if fp["ram_total_bytes"] is None:
        fp["ram_total_bytes"] = _ram_total_fallback()
    return fp


_STALENESS_KEYS = ("machine", "cpu_brand", "cpu_count_logical", "ram_total_bytes")


def hardware_matches(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Return True if the stored and current fingerprints match on key fields.

    Compares machine architecture, CPU brand, logical core count, and
    total RAM — the fields most likely to affect sampling baseline times.
    """
    return all(stored.get(k) == current.get(k) for k in _STALENESS_KEYS)
