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


def test_runner_error_carries_stage_and_detail() -> None:
    detail = RunnerErrorDetail(code="TEST", message="test error")
    err = RunnerError("predict", detail)
    assert err.stage == "predict"
    assert err.detail.code == "TEST"
