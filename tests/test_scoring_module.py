import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.scoring import (
    ContestSpec,
    evaluate_estimator,
    make_contest,
)


def test_contest_spec_validates() -> None:
    spec = ContestSpec(width=8, depth=2, n_mlps=2, flop_budget=1_000_000, ground_truth_samples=1000)
    spec.validate()


def test_contest_spec_rejects_zero_width() -> None:
    with pytest.raises(ValueError, match="width"):
        ContestSpec(
            width=0, depth=2, n_mlps=2, flop_budget=1_000_000, ground_truth_samples=1000
        ).validate()


def test_make_contest_produces_valid_data() -> None:
    spec = ContestSpec(width=8, depth=2, n_mlps=3, flop_budget=1_000_000, ground_truth_samples=200)
    data = make_contest(spec)
    assert len(data.mlps) == 3
    assert len(data.all_layer_targets) == 3
    assert len(data.final_targets) == 3
    assert len(data.avg_variances) == 3
    for targets in data.all_layer_targets:
        assert targets.shape == (2, 8)
    for final in data.final_targets:
        assert final.shape == (8,)
    for var in data.avg_variances:
        assert var >= 0.0


def test_evaluate_estimator_with_zeros_estimator() -> None:
    """An estimator that always returns zeros should produce a finite score."""
    from network_estimation.sdk import BaseEstimator

    class ZerosEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            return np.zeros((mlp.depth, mlp.width), dtype=np.float32)

    spec = ContestSpec(width=8, depth=2, n_mlps=2, flop_budget=100_000_000, ground_truth_samples=200)
    data = make_contest(spec)
    result = evaluate_estimator(ZerosEstimator(), data)
    assert isinstance(result, dict)
    assert "primary_score" in result
    assert "secondary_score" in result
    assert np.isfinite(result["primary_score"])
    assert np.isfinite(result["secondary_score"])


def test_validate_predictions_rejects_wrong_shape() -> None:
    from network_estimation.scoring import validate_predictions

    with pytest.raises(ValueError, match="shape"):
        validate_predictions(np.zeros((3, 4), dtype=np.float32), depth=2, width=4)


def test_validate_predictions_rejects_nonfinite() -> None:
    from network_estimation.scoring import validate_predictions

    arr = np.zeros((2, 4), dtype=np.float32)
    arr[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(arr, depth=2, width=4)


def test_evaluate_estimator_records_flops_used() -> None:
    """Each per-mlp record should include flops_used."""
    from network_estimation.sdk import BaseEstimator

    class ZerosEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            return np.zeros((mlp.depth, mlp.width), dtype=np.float32)

    spec = ContestSpec(width=8, depth=2, n_mlps=1, flop_budget=100_000_000, ground_truth_samples=200)
    data = make_contest(spec)
    result = evaluate_estimator(ZerosEstimator(), data)
    assert "flops_used" in result["per_mlp"][0]
