"""Helpers for validating streamed depth-row estimator outputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_depth_row(row: object, *, width: int, depth_index: int) -> NDArray[np.float32]:
    """Validate one streamed estimator row and return a float32 vector."""
    arr = np.asarray(row, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != width:
        raise ValueError(
            f"Estimator row at depth {depth_index} must have shape ({width},), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Estimator row at depth {depth_index} must contain finite values.")
    return arr
