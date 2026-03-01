from __future__ import annotations

import pytest

from circuit_estimation.runner import PredictOutcome, ResourceLimits


def test_predict_outcome_supports_required_status_and_metrics_fields() -> None:
    outcome = PredictOutcome(
        predictions=None,
        wall_time_s=0.12,
        cpu_time_s=0.08,
        rss_bytes=1024,
        peak_rss_bytes=2048,
        status="timeout",
        error_message="predict exceeded timeout",
    )

    assert outcome.status == "timeout"
    assert outcome.wall_time_s == pytest.approx(0.12)
    assert outcome.cpu_time_s == pytest.approx(0.08)
    assert outcome.rss_bytes == 1024
    assert outcome.peak_rss_bytes == 2048
    assert outcome.error_message == "predict exceeded timeout"


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
