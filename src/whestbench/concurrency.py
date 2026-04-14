# src/whestbench/concurrency.py
"""CPU thread-limiting utilities for whest/BLAS.

Provides a single function to cap the number of CPU threads used by
BLAS libraries (OpenBLAS, MKL, Accelerate) that underpin whest.

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

# Environment variable names that control BLAS thread pools.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def apply_thread_limit(n: Optional[int] = None) -> Optional[int]:
    """Cap CPU parallelism for BLAS backends.

    Sets environment variables for libraries not yet imported, and uses
    ``threadpoolctl`` for BLAS to apply the limit to libraries already loaded.

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

    # Use threadpoolctl to set BLAS thread count via the C API.
    # This works even after numpy/OpenBLAS has been imported, unlike
    # environment variables which are only read at library load time.
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=n, user_api="blas")
    except ImportError:
        pass

    return n
