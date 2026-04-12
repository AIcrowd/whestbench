# src/whestbench/concurrency.py
"""CPU thread-limiting utilities.

Provides a single function to cap the number of CPU threads used by all
numerical backends (BLAS via OpenBLAS/MKL, Numba, PyTorch, JAX/XLA).

The limit can be set in two ways (in priority order):

1. **Programmatically** — call :func:`apply_thread_limit` before importing
   any backend.
2. **Environment variable** — set ``WHEST_MAX_THREADS`` before launching
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

    Sets environment variables for libraries not yet imported, and uses
    runtime APIs (PyTorch ``set_num_threads``, ``threadpoolctl`` for
    BLAS) to apply the limit to libraries already loaded.

    Args:
        n: Maximum number of threads.  When ``None``, the value of
           ``WHEST_MAX_THREADS`` is used.  If that is also unset,
           this function is a no-op and returns ``None``.

    Returns:
        The effective thread limit that was applied, or ``None`` if no
        limit was set.
    """
    if n is None:
        env_val = os.environ.get("WHEST_MAX_THREADS")
        if env_val is None:
            return None
        n = int(env_val)

    s = str(n)
    for var in _THREAD_ENV_VARS:
        os.environ[var] = s

    # JAX/XLA: only set flags known to be valid in current XLA versions.
    # Note: --xla_intra_op_parallelism_threads was removed in newer XLA
    # and causes a fatal abort if set.
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"

    # If PyTorch is already loaded, apply the runtime cap too.
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass

    # Use threadpoolctl to set BLAS thread count via the C API.
    # This works even after numpy/OpenBLAS has been imported, unlike
    # environment variables which are only read at library load time.
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=n, user_api="blas")
    except ImportError:
        pass

    return n
