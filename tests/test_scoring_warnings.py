from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
import pytest

import whestbench
from whestbench.domain import MLP
from whestbench.scoring import (
    BudgetExhaustionWarning,
    ContestData,
    ContestSpec,
    ScoringExhaustionWarning,
    TimeExhaustionWarning,
    evaluate_estimator,
)
from whestbench.sdk import BaseEstimator


def test_warning_class_hierarchy() -> None:
    """ScoringExhaustionWarning is a UserWarning; budget/time inherit from it."""
    assert issubclass(ScoringExhaustionWarning, UserWarning)
    assert issubclass(BudgetExhaustionWarning, ScoringExhaustionWarning)
    assert issubclass(TimeExhaustionWarning, ScoringExhaustionWarning)


def test_warning_classes_reexported_from_package() -> None:
    """Library users can import the classes from the top-level package."""
    assert whestbench.ScoringExhaustionWarning is ScoringExhaustionWarning
    assert whestbench.BudgetExhaustionWarning is BudgetExhaustionWarning
    assert whestbench.TimeExhaustionWarning is TimeExhaustionWarning


def _make_tiny_data(width: int = 4, depth: int = 2, n_mlps: int = 2) -> ContestData:
    """Build minimal ContestData for `n_mlps` trivially-tiny MLPs."""
    spec = ContestSpec(
        width=width,
        depth=depth,
        n_mlps=n_mlps,
        flop_budget=1_000,
        ground_truth_samples=10,
        wall_time_limit_s=5.0,
        residual_wall_time_limit_s=5.0,
    )
    weights = [fnp.zeros((width, width), dtype=fnp.float32) for _ in range(depth)]
    mlps = [MLP(width=width, depth=depth, weights=weights) for _ in range(n_mlps)]
    target = fnp.zeros((depth, width), dtype=fnp.float32)
    return ContestData(
        spec=spec,
        mlps=mlps,
        all_layer_targets=[target for _ in range(n_mlps)],
        final_targets=[target[-1] for _ in range(n_mlps)],
        avg_variances=[0.0 for _ in range(n_mlps)],
    )


class _HungryEstimator(BaseEstimator):
    """Estimator that always exhausts the FLOP budget immediately."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        raise flops.BudgetExhaustedError("test", flop_cost=budget + 1, flops_remaining=0)


def test_budget_exhaustion_emits_warning() -> None:
    """evaluate_estimator emits BudgetExhaustionWarning once per exhausted MLP."""
    data = _make_tiny_data(n_mlps=2)
    with pytest.warns(BudgetExhaustionWarning) as records:
        result = evaluate_estimator(_HungryEstimator(), data)

    assert len(records) == 2
    msgs = [str(r.message) for r in records]
    assert all("exhausted FLOP budget" in m for m in msgs)
    assert all("estimator output set to zeros" in m for m in msgs)
    # The two warnings reference MLP 0 and MLP 1 respectively.
    assert any("MLP 0" in m for m in msgs)
    assert any("MLP 1" in m for m in msgs)
    # All MLPs are flagged exhausted.
    for entry in result["per_mlp"]:
        assert entry["budget_exhausted"] is True


def test_traceback_in_per_mlp_entry_on_budget_exhaustion() -> None:
    """The per-MLP entry stores the BudgetExhaustedError traceback as a string."""
    data = _make_tiny_data(n_mlps=2)
    with pytest.warns(BudgetExhaustionWarning):
        result = evaluate_estimator(_HungryEstimator(), data)

    for entry in result["per_mlp"]:
        assert entry["budget_exhausted"] is True
        assert isinstance(entry["traceback"], str)
        assert "BudgetExhaustedError" in entry["traceback"]


def test_fail_fast_reraises_budget_exhausted() -> None:
    """Under fail_fast=True, BudgetExhaustedError propagates instead of being captured."""
    data = _make_tiny_data(n_mlps=2)
    with pytest.raises(flops.BudgetExhaustedError):
        evaluate_estimator(_HungryEstimator(), data, fail_fast=True)
