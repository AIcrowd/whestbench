"""Post-hoc combined budget check: C_m > B_m → zero pred + multiplier = 1.0."""

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


def _make_data(width: int = 4, depth: int = 2, flop_budget: int = 10_000_000_000) -> ContestData:
    weights = [fnp.array(fnp.zeros((width, width), dtype=fnp.float32)) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    final_target = fnp.array([1.0, 2.0, 3.0, 4.0], dtype=fnp.float32)
    all_target = fnp.array([[0.5, 1.0, 1.5, 2.0], [1.0, 2.0, 3.0, 4.0]], dtype=fnp.float32)
    return ContestData(
        spec=ContestSpec(
            width=width,
            depth=depth,
            n_mlps=1,
            flop_budget=flop_budget,
            ground_truth_samples=100,
            residual_wall_time_limit_s=None,  # Disable mid-call cap to test the post-hoc check.
        ),
        mlps=[mlp],
        all_layer_targets=[all_target],
        final_targets=[final_target],
        avg_variances=[0.0],
    )


def test_combined_budget_exhausted_when_c_m_exceeds_b_m():
    """Sum of FLOPs and lambda*residual exceeding B_m triggers post-hoc zero-out."""
    flop_budget = 10_000_000_000  # 1e10
    # Pick F = 0.7 B_m, R s.t. lambda*R = 0.5 B_m. Sum = 1.2 B_m > B_m → over.
    f_m = int(0.7 * flop_budget)  # 7e9
    r_m = (0.5 * flop_budget) / LAMBDA_FLOPS_PER_SECOND  # 0.5e10 / 1e10 = 0.5 sec
    estimator = _StatsEstimator(
        PredictStats(
            flops_used=f_m,
            wall_time_s=r_m,
            flopscope_backend_time_s=0.0,
            flopscope_overhead_time_s=0.0,
            residual_wall_time_s=r_m,
        )
    )
    data = _make_data(flop_budget=flop_budget)
    result = evaluate_estimator(estimator, data)
    per_mlp = result["per_mlp"][0]

    assert per_mlp["combined_budget_exhausted"] is True
    # Failure → s_m = MSE(0, Y) * 1.0
    EXPECTED_ZERO_PRED_MSE = 7.5
    assert per_mlp["adjusted_final_layer_mse"] == pytest.approx(EXPECTED_ZERO_PRED_MSE, abs=1e-5)


def test_combined_budget_not_triggered_when_c_m_within_budget():
    """C_m <= B_m must NOT trigger combined_budget_exhausted."""
    flop_budget = 10_000_000_000
    f_m = int(0.5 * flop_budget)
    r_m = (0.3 * flop_budget) / LAMBDA_FLOPS_PER_SECOND  # 0.3 sec
    # C_m = 0.5 + 0.3 = 0.8 B_m, within budget.
    estimator = _StatsEstimator(
        PredictStats(
            flops_used=f_m,
            wall_time_s=r_m,
            flopscope_backend_time_s=0.0,
            flopscope_overhead_time_s=0.0,
            residual_wall_time_s=r_m,
        )
    )
    data = _make_data(flop_budget=flop_budget)
    result = evaluate_estimator(estimator, data)
    per_mlp = result["per_mlp"][0]

    assert per_mlp["combined_budget_exhausted"] is False
    # Valid run → multiplier = max(0.5, 0.8) = 0.8
    assert per_mlp["adjusted_final_layer_mse"] == pytest.approx(7.5 * 0.8, abs=1e-5)
