"""Verify scoring consumes worker-reported timing under subprocess mode."""

from __future__ import annotations

from typing import Optional

import flopscope.numpy as fnp
import pytest

from whestbench.domain import MLP
from whestbench.runner import PredictStats
from whestbench.scoring import ContestData, ContestSpec, evaluate_estimator
from whestbench.sdk import BaseEstimator


class _FakeRunnerEstimator(BaseEstimator):
    """Simulates a subprocess runner: predict returns zeros and reports its own stats."""

    def __init__(self, stats: PredictStats) -> None:
        self._stats = stats

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return fnp.zeros((mlp.depth, mlp.width))

    def last_predict_stats(self) -> Optional[PredictStats]:
        return self._stats


def _make_contest_data(width: int = 4, depth: int = 2) -> ContestData:
    spec = ContestSpec(
        width=width,
        depth=depth,
        n_mlps=1,
        flop_budget=10_000_000_000,
        ground_truth_samples=100,
        residual_wall_time_limit_s=1.0,
    )
    weights = [fnp.array(fnp.zeros((width, width), dtype=fnp.float32)) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    target = fnp.zeros((depth, width), dtype=fnp.float32)
    return ContestData(
        spec=spec,
        mlps=[mlp],
        all_layer_targets=[target],
        final_targets=[target[-1]],
        avg_variances=[0.0],
    )


def test_scoring_uses_worker_reported_residual_wall_time():
    """When estimator exposes last_predict_stats with non-zero residual, scoring uses it."""
    data = _make_contest_data()
    estimator = _FakeRunnerEstimator(
        PredictStats(
            flops_used=100,
            wall_time_s=2.5,
            flopscope_backend_time_s=0.1,
            flopscope_overhead_time_s=0.05,
            residual_wall_time_s=2.35,  # Above residual_wall_time_limit_s=1.0
        )
    )
    result = evaluate_estimator(estimator, data)
    per_mlp = result["per_mlp"][0]
    assert per_mlp["residual_wall_time_s"] == pytest.approx(2.35)
    assert per_mlp["wall_time_s"] == pytest.approx(2.5)
    assert per_mlp["residual_wall_time_exhausted"] is True


def test_scoring_falls_back_to_budget_ctx_when_no_stats():
    """When estimator has no last_predict_stats, scoring uses host budget_ctx values."""
    data = _make_contest_data()

    class _LocalEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
            return fnp.zeros((mlp.depth, mlp.width))

    estimator = _LocalEstimator()
    result = evaluate_estimator(estimator, data)
    per_mlp = result["per_mlp"][0]
    # No stats reported; scoring falls back to budget_ctx which sees roughly 0 residual
    # for a trivial-zero estimator. Just assert the field is present and finite.
    assert "residual_wall_time_s" in per_mlp
    assert per_mlp["residual_wall_time_s"] >= 0.0
    assert per_mlp["residual_wall_time_exhausted"] is False
