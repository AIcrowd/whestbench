from __future__ import annotations

import numpy as np
import pytest

from network_estimation.scoring import (
    ContestSpec,
    evaluate_estimator,
    make_contest,
    validate_predictions,
)
from network_estimation.estimators import MeanPropagationEstimator
from network_estimation.sdk import BaseEstimator
from network_estimation.domain import MLP


def test_validate_predictions_accepts_correct_shape() -> None:
    arr = np.ones((3, 4), dtype=np.float32)
    result = validate_predictions(arr, depth=3, width=4)
    np.testing.assert_array_equal(result, arr)


def test_validate_predictions_rejects_wrong_shape() -> None:
    arr = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        validate_predictions(arr, depth=3, width=4)


def test_validate_predictions_rejects_non_finite() -> None:
    arr = np.ones((3, 4), dtype=np.float32)
    arr[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(arr, depth=3, width=4)


def test_make_contest_produces_correct_data() -> None:
    spec = ContestSpec(width=4, depth=2, n_mlps=3, estimator_budget=100, ground_truth_budget=200)
    data = make_contest(spec)
    assert len(data.mlps) == 3
    assert len(data.all_layer_targets) == 3
    assert len(data.final_targets) == 3
    assert len(data.avg_variances) == 3
    for mlp in data.mlps:
        assert mlp.width == 4
        assert mlp.depth == 2


def test_evaluate_estimator_returns_scores() -> None:
    spec = ContestSpec(width=4, depth=2, n_mlps=2, estimator_budget=100, ground_truth_budget=200)
    data = make_contest(spec)
    estimator = MeanPropagationEstimator()
    result = evaluate_estimator(estimator, data)
    assert "primary_score" in result
    assert "secondary_score" in result
    assert "per_mlp" in result
    assert isinstance(result["primary_score"], float)
    assert isinstance(result["secondary_score"], float)


def test_evaluate_estimator_handles_error_gracefully() -> None:
    spec = ContestSpec(width=4, depth=2, n_mlps=1, estimator_budget=100, ground_truth_budget=200)
    data = make_contest(spec)

    class _BadEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> np.ndarray:
            raise RuntimeError("intentional error")

    result = evaluate_estimator(_BadEstimator(), data)
    assert "primary_score" in result
    assert result["per_mlp"][0].get("error") is not None
