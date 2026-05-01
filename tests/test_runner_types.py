import pytest

from whestbench.runner import (
    ResourceLimits,
    RunnerError,
    RunnerErrorDetail,
)


def test_resource_limits_rejects_nonpositive_setup_timeout() -> None:
    with pytest.raises(ValueError):
        ResourceLimits(
            setup_timeout_s=0, predict_timeout_s=1.0, memory_limit_mb=1024, flop_budget=1000
        )


def test_resource_limits_time_limits_default_none() -> None:
    limits = ResourceLimits(
        setup_timeout_s=5.0, predict_timeout_s=30.0, memory_limit_mb=4096, flop_budget=100_000_000
    )
    assert limits.wall_time_limit_s is None
    assert limits.residual_wall_time_limit_s is None


def test_resource_limits_accepts_time_limits() -> None:
    limits = ResourceLimits(
        setup_timeout_s=5.0,
        predict_timeout_s=30.0,
        memory_limit_mb=4096,
        flop_budget=100_000_000,
        wall_time_limit_s=10.0,
        residual_wall_time_limit_s=5.0,
    )
    assert limits.wall_time_limit_s == 10.0
    assert limits.residual_wall_time_limit_s == 5.0


def test_resource_limits_rejects_nonpositive_wall_time_limit() -> None:
    with pytest.raises(ValueError):
        ResourceLimits(
            setup_timeout_s=5.0,
            predict_timeout_s=30.0,
            memory_limit_mb=4096,
            flop_budget=100_000_000,
            wall_time_limit_s=0.0,
        )


def test_resource_limits_rejects_nonpositive_residual_wall_time_limit() -> None:
    with pytest.raises(ValueError):
        ResourceLimits(
            setup_timeout_s=5.0,
            predict_timeout_s=30.0,
            memory_limit_mb=4096,
            flop_budget=100_000_000,
            residual_wall_time_limit_s=-1.0,
        )


def test_runner_error_carries_stage_and_detail() -> None:
    detail = RunnerErrorDetail(code="TEST", message="test error")
    err = RunnerError("predict", detail)
    assert err.stage == "predict"
    assert err.detail.code == "TEST"
