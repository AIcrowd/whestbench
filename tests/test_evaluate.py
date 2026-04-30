from __future__ import annotations

import flopscope.numpy as fnp
import pytest

from whestbench.domain import MLP
from whestbench.scoring import (
    ContestSpec,
    evaluate_estimator,
    make_contest,
    validate_predictions,
)
from whestbench.sdk import BaseEstimator


def test_contest_spec_has_flop_budget() -> None:
    spec = ContestSpec(
        width=4,
        depth=2,
        n_mlps=1,
        flop_budget=1_000_000,
        ground_truth_samples=200,
    )
    assert spec.flop_budget == 1_000_000


def test_validate_predictions_accepts_correct_shape() -> None:
    arr = fnp.ones((3, 4))
    result = validate_predictions(arr, depth=3, width=4)
    assert tuple(result.shape) == (3, 4)


def test_validate_predictions_rejects_wrong_shape() -> None:
    arr = fnp.ones((2, 4))
    with pytest.raises(ValueError, match="shape") as exc_info:
        validate_predictions(arr, depth=3, width=4)
    details = getattr(exc_info.value, "details", None)
    assert isinstance(details, dict)
    assert details.get("expected_shape") == [3, 4]
    assert details.get("got_shape") == [2, 4]


def test_validate_predictions_rejects_non_finite() -> None:
    arr = fnp.ones((3, 4), dtype=fnp.float32)
    arr[0, 0] = float("nan")
    arr = fnp.array(arr)
    with pytest.raises(ValueError, match="finite") as exc_info:
        validate_predictions(arr, depth=3, width=4)
    details = getattr(exc_info.value, "details", None)
    assert isinstance(details, dict)
    assert details.get("expected_shape") == [3, 4]


def test_make_contest_produces_correct_data() -> None:
    spec = ContestSpec(
        width=4,
        depth=2,
        n_mlps=3,
        flop_budget=1_000_000,
        ground_truth_samples=200,
    )
    data = make_contest(spec)
    assert len(data.mlps) == 3
    assert len(data.all_layer_targets) == 3
    assert len(data.final_targets) == 3
    assert len(data.avg_variances) == 3


def test_evaluate_estimator_returns_mse_scores() -> None:
    spec = ContestSpec(
        width=4,
        depth=2,
        n_mlps=2,
        flop_budget=100_000_000,
        ground_truth_samples=200,
    )
    data = make_contest(spec)

    class _SimpleEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
            return fnp.zeros((mlp.depth, mlp.width))

    estimator = _SimpleEstimator()
    result = evaluate_estimator(estimator, data)
    assert "primary_score" in result
    assert "secondary_score" in result
    assert "per_mlp" in result
    assert isinstance(result["primary_score"], float)
    assert result["primary_score"] >= 0.0
    assert "flops_used" in result["per_mlp"][0]


def test_evaluate_estimator_handles_error_gracefully() -> None:
    spec = ContestSpec(
        width=4,
        depth=2,
        n_mlps=1,
        flop_budget=100_000_000,
        ground_truth_samples=200,
    )
    data = make_contest(spec)

    class _BadEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
            raise RuntimeError("intentional error")

    result = evaluate_estimator(_BadEstimator(), data)
    assert "primary_score" in result
    assert result["per_mlp"][0].get("error") is not None
