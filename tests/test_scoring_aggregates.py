"""New suite-level diagnostic aggregates emitted by evaluate_estimator."""

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


class _StatsEstimator(BaseEstimator):
    """Returns zeros and reports pre-fabricated stats."""

    def __init__(self, stats: PredictStats) -> None:
        self._stats = stats

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return fnp.zeros((mlp.depth, mlp.width))

    def last_predict_stats(self) -> Optional[PredictStats]:
        return self._stats


def _make_data(
    n_mlps: int = 1, width: int = 4, depth: int = 2, flop_budget: int = 10_000_000_000
) -> ContestData:
    mlps = [
        MLP(
            width=width,
            depth=depth,
            weights=[fnp.array(fnp.zeros((width, width), dtype=fnp.float32)) for _ in range(depth)],
        )
        for _ in range(n_mlps)
    ]
    final_t = fnp.array([1.0, 2.0, 3.0, 4.0], dtype=fnp.float32)
    all_t = fnp.array([[0.5, 1.0, 1.5, 2.0], [1.0, 2.0, 3.0, 4.0]], dtype=fnp.float32)
    return ContestData(
        spec=ContestSpec(
            width=width,
            depth=depth,
            n_mlps=n_mlps,
            flop_budget=flop_budget,
            ground_truth_samples=100,
        ),
        mlps=mlps,
        all_layer_targets=[all_t] * n_mlps,
        final_targets=[final_t] * n_mlps,
        avg_variances=[0.0] * n_mlps,
    )


def test_best_and_worst_mlp_adjusted_final_layer_mse_present():
    data = _make_data()
    estimator = _StatsEstimator(
        PredictStats(
            flops_used=0,
            wall_time_s=0.0,
            flopscope_backend_time_s=0.0,
            flopscope_overhead_time_s=0.0,
            residual_wall_time_s=0.0,
        )
    )
    result = evaluate_estimator(estimator, data)
    s_m = result["per_mlp"][0]["adjusted_final_layer_mse"]
    assert result["best_mlp_adjusted_final_layer_mse"] == pytest.approx(s_m)
    assert result["worst_mlp_adjusted_final_layer_mse"] == pytest.approx(s_m)


def test_mean_score_multiplier_at_floor_for_zero_compute():
    """C_m = 0 → multiplier clamped to 0.1."""
    data = _make_data()
    estimator = _StatsEstimator(
        PredictStats(
            flops_used=0,
            wall_time_s=0.0,
            flopscope_backend_time_s=0.0,
            flopscope_overhead_time_s=0.0,
            residual_wall_time_s=0.0,
        )
    )
    result = evaluate_estimator(estimator, data)
    assert result["mean_score_multiplier"] == pytest.approx(0.1)


def test_mean_compute_utilization_unclamped():
    """Compute utilization is unclamped: 80% budget use → 0.80."""
    flop_budget = 10_000_000_000
    f_used = int(0.8 * flop_budget)
    data = _make_data(flop_budget=flop_budget)
    estimator = _StatsEstimator(
        PredictStats(
            flops_used=f_used,
            wall_time_s=0.0,
            flopscope_backend_time_s=0.0,
            flopscope_overhead_time_s=0.0,
            residual_wall_time_s=0.0,
        )
    )
    result = evaluate_estimator(estimator, data)
    assert result["mean_compute_utilization"] == pytest.approx(0.8)


def test_n_failed_mlps_counts_all_flag_types():
    """Any failure flag or error_code counts toward n_failed_mlps."""
    data = _make_data(n_mlps=2)

    call_count = [0]

    class _MixedEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            call_count[0] += 1
            if call_count[0] == 1:
                return fnp.zeros((mlp.depth, mlp.width))
            raise RuntimeError("MLP 1 failed")

    result = evaluate_estimator(_MixedEstimator(), data)
    assert result["n_failed_mlps"] == 1


def test_failure_breakdown_counts_per_flag():
    """failure_breakdown reports independent counts per failure flag."""
    data = _make_data(n_mlps=1)

    class _Boom(BaseEstimator):
        def predict(self, mlp, budget):
            raise RuntimeError("crash")

    result = evaluate_estimator(_Boom(), data)
    bd = result["failure_breakdown"]
    assert bd["error"] == 1
    # No exhaustion flags fired because predict raised before BudgetContext caught anything.
    assert bd["budget_exhausted"] == 0
    assert bd["time_exhausted"] == 0
    assert bd["residual_wall_time_exhausted"] == 0
    assert bd["combined_budget_exhausted"] == 0


def test_mean_effective_compute_matches_per_mlp_mean():
    """mean_effective_compute equals the arithmetic mean of per-MLP effective_compute."""
    data = _make_data(n_mlps=2)

    class _Z(BaseEstimator):
        def predict(self, mlp, budget):
            return fnp.zeros((mlp.depth, mlp.width))

    result = evaluate_estimator(_Z(), data)
    pm = result["per_mlp"]
    expected = sum(e["effective_compute"] for e in pm) / len(pm)
    assert result["mean_effective_compute"] == pytest.approx(expected)
