# src/network_estimation/concurrency.py
"""CPU thread-limiting utilities.

Provides a single function to cap the number of CPU threads used by all
numerical backends (BLAS via OpenBLAS/MKL, Numba, PyTorch, JAX/XLA).

The limit can be set in two ways (in priority order):

1. **Programmatically** — call :func:`apply_thread_limit` before importing
   any backend.
2. **Environment variable** — set ``NESTIM_MAX_THREADS`` before launching
   the process.  This is picked up automatically by
   :func:`apply_thread_limit` when no explicit *n* is passed.

The ``--max-threads`` CLI flag (available on ``profile-simulation``,
``run``, ``create-dataset``, and ``smoke-test``) calls this function
early, before any backend module is imported.
"""
from __future__ import annotations

import os
from typing import Optional

# Environment variable names that control thread pools in common
# numerical libraries.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def apply_thread_limit(n: Optional[int] = None) -> Optional[int]:
    """Cap CPU parallelism for all numerical backends.

    Must be called **before** backend modules are imported so that
    libraries pick up the environment variables at initialisation time.
    If a library (e.g. PyTorch) is already imported, the runtime knob
    is set as well.

    Args:
        n: Maximum number of threads.  When ``None``, the value of
           ``NESTIM_MAX_THREADS`` is used.  If that is also unset,
           this function is a no-op and returns ``None``.

    Returns:
        The effective thread limit that was applied, or ``None`` if no
        limit was set.
    """
    if n is None:
        env_val = os.environ.get("NESTIM_MAX_THREADS")
        if env_val is None:
            return None
        n = int(env_val)

    s = str(n)
    for var in _THREAD_ENV_VARS:
        os.environ[var] = s

    # JAX/XLA uses a flag-style variable.
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true "
        f"--xla_intra_op_parallelism_threads={n}"
    )

    # If PyTorch is already loaded, apply the runtime cap too.
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass

    return n
