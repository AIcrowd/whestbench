"""Version + runtime-state lookups for dataset metadata.

These are embedded in ``metadata.json`` under top-level version pins and the
``bake_config`` subobject so consumers know which whestbench/flopscope/torch
+ which determinism settings produced a dataset. Required for bit-exact
reproduction across hosts and over time — see the `bit-equivalence requirements`
section of ``docs/how-to/parallel-bake.md``.

All functions degrade gracefully when something can't be determined: they return
``"unknown"`` or ``None`` rather than raising.
"""

from __future__ import annotations

import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional


def whestbench_version() -> str:
    """Installed whestbench package version, or ``"unknown"``."""
    try:
        return version("whestbench")
    except PackageNotFoundError:
        return "unknown"


def flopscope_version() -> str:
    """Installed flopscope package version, or ``"unknown"``.

    Captured because all weight init goes through ``flopscope.numpy``; a
    different flopscope version could change weight values for the same seed.
    """
    try:
        return version("flopscope")
    except PackageNotFoundError:
        return "unknown"


def nvidia_driver_version() -> Optional[str]:
    """Best-effort read of NVIDIA driver version via ``nvidia-smi``.

    Returns ``None`` if ``nvidia-smi`` is unavailable or the call fails.
    Captured because kernel implementations can differ between drivers even
    when the torch+CUDA toolkit version is held constant.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip().splitlines()
    if not out:
        return None
    return out[0].strip() or None


def torch_determinism_state() -> Dict[str, Any]:
    """Snapshot torch's current determinism-relevant flags + env state.

    Reads what's CURRENTLY in effect — captures the actual state during the bake,
    rather than what whestbench requires. Returns a JSON-serialisable dict for
    direct inclusion in ``metadata.json``.

    Importantly, ``cublas_workspace_config`` reflects the ``CUBLAS_WORKSPACE_CONFIG``
    env var, NOT a torch property — torch itself doesn't track it. The env var
    must be set BEFORE cuda init for cuBLAS deterministic behavior, so
    "what's in os.environ at bake time" is the most useful snapshot.
    """
    import torch  # local import keeps non-torch callers (CPU path) free of torch dep

    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
