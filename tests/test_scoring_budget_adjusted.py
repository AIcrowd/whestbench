"""Parameterized tests for the budget-adjusted per-MLP score formula."""

from __future__ import annotations

from typing import Optional

import flopscope.numpy as fnp
import pytest

from whestbench.domain import MLP
from whestbench.runner import PredictStats
from whestbench.scoring import (
    ContestData,
    ContestSpec,
    evaluate_estimator,
)
from whestbench.sdk import BaseEstimator


class _ZeroPredEstimator(BaseEstimator):
    """Returns zeros so MSE_final equals mean-square of final_target."""

    def __init__(self, stats: PredictStats) -> None:
        self._stats = stats

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return fnp.zeros((mlp.depth, mlp.width))

    def last_predict_stats(self) -> Optional[PredictStats]:
        return self._stats


def _make_data_with_nontrivial_target(
    width: int = 4, depth: int = 2, flop_budget: int = 10_000_000_000
) -> ContestData:
    """Construct contest data where the final target has a known mean-square."""
    weights = [fnp.array(fnp.zeros((width, width), dtype=fnp.float32)) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    # Non-zero target so MSE(0, target) = mean(target**2) is positive.
    final_target = fnp.array(
        [1.0, 2.0, 3.0, 4.0], dtype=fnp.float32
    )  # mean(squared) = (1+4+9+16)/4 = 7.5
    all_target = fnp.array([[0.5, 1.0, 1.5, 2.0], [1.0, 2.0, 3.0, 4.0]], dtype=fnp.float32)
    return ContestData(
        spec=ContestSpec(
            width=width,
            depth=depth,
            n_mlps=1,
            flop_budget=flop_budget,
            ground_truth_samples=100,
        ),
        mlps=[mlp],
        all_layer_targets=[all_target],
        final_targets=[final_target],
        avg_variances=[0.0],
    )


@pytest.mark.parametrize(
    "ratio,expected_multiplier",
    [
        (0.0, 0.5),  # below floor → clamped to 0.5
        (0.3, 0.5),  # below floor
        (0.5, 0.5),  # at floor
        (0.6, 0.6),  # above floor
        (0.8, 0.8),  # well within budget
        (1.0, 1.0),  # at budget
    ],
)
def test_s_m_applies_max_floor_multiplier(ratio: float, expected_multiplier: float):
    """For valid runs, s_m = MSE_final * max(0.5, C_m / B_m)."""
    flop_budget = 10_000_000_000
    target_c_m = int(flop_budget * ratio)
    data = _make_data_with_nontrivial_target(flop_budget=flop_budget)

    # Configure the estimator to report exactly C_m via pure FLOPs (zero residual).
    estimator = _ZeroPredEstimator(
        PredictStats(
            flops_used=target_c_m,
            wall_time_s=0.0,
            flopscope_backend_time_s=0.0,
            flopscope_overhead_time_s=0.0,
            residual_wall_time_s=0.0,
        )
    )
    result = evaluate_estimator(estimator, data)
    per_mlp = result["per_mlp"][0]

    expected_mse_final = 7.5  # mean of target_final**2 with prediction = 0
    expected_s_m = expected_mse_final * expected_multiplier

    assert per_mlp["final_layer_mse"] == pytest.approx(expected_mse_final, abs=1e-5)
    assert per_mlp["adjusted_final_layer_mse"] == pytest.approx(expected_s_m, abs=1e-5)
    # Aggregate primary_score is now the budget-adjusted suite mean.
    assert result["primary_score"] == pytest.approx(expected_s_m, abs=1e-5)


def test_primary_score_is_budget_adjusted_not_raw_mse():
    """primary_score must be the suite mean of s_m, not raw final_mse."""
    flop_budget = 10_000_000_000
    data = _make_data_with_nontrivial_target(flop_budget=flop_budget)
    # Use 30% of the budget → multiplier should hit the 0.5 floor.
    estimator = _ZeroPredEstimator(
        PredictStats(
            flops_used=int(flop_budget * 0.3),
            wall_time_s=0.0,
            flopscope_backend_time_s=0.0,
            flopscope_overhead_time_s=0.0,
            residual_wall_time_s=0.0,
        )
    )
    result = evaluate_estimator(estimator, data)
    raw_mse = result["per_mlp"][0]["final_layer_mse"]
    primary = result["primary_score"]
    # Below the floor → multiplier = 0.5, so primary should be raw / 2.
    assert primary == pytest.approx(raw_mse * 0.5, abs=1e-5)
    assert primary < raw_mse  # Verifies discount is being applied.
