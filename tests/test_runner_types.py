import numpy as np
import pytest

from circuit_estimation.runner import DepthRowOutcome, ResourceLimits


def test_resource_limits_require_setup_and_predict_caps() -> None:
    with pytest.raises(ValueError, match="setup_timeout_s"):
        ResourceLimits(
            setup_timeout_s=0.0,
            predict_timeout_s=1.0,
            memory_limit_mb=128,
        )
    with pytest.raises(ValueError, match="predict_timeout_s"):
        ResourceLimits(
            setup_timeout_s=1.0,
            predict_timeout_s=-1.0,
            memory_limit_mb=128,
        )
    with pytest.raises(ValueError, match="memory_limit_mb"):
        ResourceLimits(
            setup_timeout_s=1.0,
            predict_timeout_s=1.0,
            memory_limit_mb=0,
        )
    with pytest.raises(ValueError, match="cpu_time_limit_s"):
        ResourceLimits(
            setup_timeout_s=1.0,
            predict_timeout_s=1.0,
            memory_limit_mb=128,
            cpu_time_limit_s=0.0,
        )

    limits = ResourceLimits(
        setup_timeout_s=2.0,
        predict_timeout_s=0.5,
        memory_limit_mb=256,
        cpu_time_limit_s=None,
    )
    assert limits.setup_timeout_s == pytest.approx(2.0)
    assert limits.predict_timeout_s == pytest.approx(0.5)
    assert limits.memory_limit_mb == 256


def test_depth_row_outcome_ok_status() -> None:
    row = np.array([0.1, -0.2], dtype=np.float32)
    outcome = DepthRowOutcome(depth_index=0, row=row, wall_time_s=0.01, status="ok")
    assert outcome.status == "ok"
    assert outcome.row is not None
    assert outcome.depth_index == 0
    assert outcome.wall_time_s == pytest.approx(0.01)
    assert outcome.error_message is None


def test_depth_row_outcome_error_status() -> None:
    outcome = DepthRowOutcome(
        depth_index=2, row=None, wall_time_s=0.5,
        status="error", error_message="boom",
    )
    assert outcome.status == "error"
    assert outcome.row is None
    assert outcome.error_message == "boom"

