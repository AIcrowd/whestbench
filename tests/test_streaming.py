from __future__ import annotations

import numpy as np
import pytest

from circuit_estimation.streaming import validate_depth_row


def test_validate_depth_row_accepts_float_vector() -> None:
    row = validate_depth_row(np.array([0.1, -0.2], dtype=np.float32), width=2, depth_index=0)
    assert row.shape == (2,)


def test_validate_depth_row_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_depth_row(np.zeros((2, 1), dtype=np.float32), width=2, depth_index=1)


def test_validate_depth_row_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_depth_row(np.array([np.nan, 0.0], dtype=np.float32), width=2, depth_index=2)
