"""Verify effective compute C_m = F_m + lambda*R_m is computed and surfaced."""

from __future__ import annotations

from typing import Optional

import flopscope.numpy as fnp
import pytest

from whestbench.domain import MLP
from whestbench.runner import PredictStats
from whestbench.scoring import (
    LAMBDA_FLOPS_PER_SECOND,
    ContestData,
    ContestSpec,
    evaluate_estimator,
)
from whestbench.sdk import BaseEstimator


class _StatsEstimator(BaseEstimator):
    def __init__(self, stats: PredictStats) -> None:
        self._stats = stats

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return fnp.zeros((mlp.depth, mlp.width))

    def last_predict_stats(self) -> Optional[PredictStats]:
        return self._stats


def _make_data(width: int = 4, depth: int = 2) -> ContestData:
    weights = [fnp.array(fnp.zeros((width, width), dtype=fnp.float32)) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    target = fnp.zeros((depth, width), dtype=fnp.float32)
    return ContestData(
        spec=ContestSpec(
            width=width,
            depth=depth,
            n_mlps=1,
            flop_budget=10_000_000_000,
            ground_truth_samples=100,
            # This module tests the PRICED regime, so name the rate explicitly. The
            # ContestSpec default is now 0 (gated), which would make C_m == F_m and
            # silently turn every assertion here into a tautology.
            lambda_flops_per_second=LAMBDA_FLOPS_PER_SECOND,
            residual_wall_time_limit_s=None,
        ),
        mlps=[mlp],
        all_layer_targets=[target],
        final_targets=[fnp.asarray(target[-1])],
        avg_variances=[0.0],
    )


def test_lambda_constant_is_1e11():
    """Module-level conversion rate must equal 10^11 FLOPs/second per the proposal."""
    assert LAMBDA_FLOPS_PER_SECOND == 1e11


def test_effective_compute_combines_flops_and_residual():
    """C_m = F_m + lambda * R_m surfaced on per-MLP record."""
    data = _make_data()
    flops_used = 1_000_000_000  # 1e9
    residual_s = 0.5
    expected_c_m = flops_used + LAMBDA_FLOPS_PER_SECOND * residual_s  # 1e9 + 5e10 = 5.1e10
    estimator = _StatsEstimator(
        PredictStats(
            flops_used=flops_used,
            wall_time_s=0.6,
            flopscope_backend_time_s=0.05,
            flopscope_overhead_time_s=0.05,
            residual_wall_time_s=residual_s,
        )
    )
    result = evaluate_estimator(estimator, data)
    per_mlp = result["per_mlp"][0]
    assert "effective_compute" in per_mlp
    assert per_mlp["effective_compute"] == pytest.approx(expected_c_m)


def test_effective_compute_with_zero_residual_equals_flops_used():
    """When residual is 0, C_m == F_m."""
    data = _make_data()
    estimator = _StatsEstimator(
        PredictStats(
            flops_used=500_000_000,
            wall_time_s=0.1,
            flopscope_backend_time_s=0.1,
            flopscope_overhead_time_s=0.0,
            residual_wall_time_s=0.0,
        )
    )
    result = evaluate_estimator(estimator, data)
    assert result["per_mlp"][0]["effective_compute"] == pytest.approx(500_000_000)


def test_effective_compute_respects_configured_lambda():
    """ContestSpec.lambda_flops_per_second flows into the surfaced C_m (single source)."""
    data = _make_data()
    data.spec.lambda_flops_per_second = 2e11  # double the default rate
    flops_used = 1_000_000_000  # 1e9
    residual_s = 0.5
    expected_c_m = flops_used + 2e11 * residual_s  # 1e9 + 1e11 = 1.01e11
    estimator = _StatsEstimator(
        PredictStats(
            flops_used=flops_used,
            wall_time_s=0.6,
            flopscope_backend_time_s=0.05,
            flopscope_overhead_time_s=0.05,
            residual_wall_time_s=residual_s,
        )
    )
    result = evaluate_estimator(estimator, data)
    assert result["per_mlp"][0]["effective_compute"] == pytest.approx(expected_c_m)
