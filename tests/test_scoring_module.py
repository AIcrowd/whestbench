import time

import pytest
import whest as we

from whestbench.scoring import (
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
    from whestbench.sdk import BaseEstimator

    class ZerosEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            return we.zeros((mlp.depth, mlp.width), dtype=we.float32)

    spec = ContestSpec(
        width=8, depth=2, n_mlps=2, flop_budget=100_000_000, ground_truth_samples=200
    )
    data = make_contest(spec)
    result = evaluate_estimator(ZerosEstimator(), data)
    assert isinstance(result, dict)
    assert "primary_score" in result
    assert "secondary_score" in result
    assert we.isfinite(result["primary_score"])
    assert we.isfinite(result["secondary_score"])


def test_validate_predictions_rejects_wrong_shape() -> None:
    from whestbench.scoring import validate_predictions

    with pytest.raises(ValueError, match="shape"):
        validate_predictions(we.zeros((3, 4), dtype=we.float32), depth=2, width=4)


def test_validate_predictions_rejects_nonfinite() -> None:
    from whestbench.scoring import validate_predictions

    arr = we.zeros((2, 4), dtype=we.float32)
    arr[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(arr, depth=2, width=4)


def test_contest_spec_time_limits_default_none() -> None:
    spec = ContestSpec(width=8, depth=2, n_mlps=2, flop_budget=1_000_000, ground_truth_samples=1000)
    assert spec.wall_time_limit_s is None
    assert spec.untracked_time_limit_s is None


def test_contest_spec_accepts_time_limits() -> None:
    spec = ContestSpec(
        width=8,
        depth=2,
        n_mlps=2,
        flop_budget=1_000_000,
        ground_truth_samples=1000,
        wall_time_limit_s=10.0,
        untracked_time_limit_s=5.0,
    )
    assert spec.wall_time_limit_s == 10.0
    assert spec.untracked_time_limit_s == 5.0
    spec.validate()


def test_contest_spec_rejects_nonpositive_wall_time_limit() -> None:
    with pytest.raises(ValueError, match="wall_time_limit_s"):
        ContestSpec(
            width=8,
            depth=2,
            n_mlps=2,
            flop_budget=1_000_000,
            ground_truth_samples=1000,
            wall_time_limit_s=0.0,
        ).validate()


def test_contest_spec_rejects_nonpositive_untracked_time_limit() -> None:
    with pytest.raises(ValueError, match="untracked_time_limit_s"):
        ContestSpec(
            width=8,
            depth=2,
            n_mlps=2,
            flop_budget=1_000_000,
            ground_truth_samples=1000,
            untracked_time_limit_s=-1.0,
        ).validate()


def test_evaluate_estimator_records_flops_used() -> None:
    """Each per-mlp record should include flops_used."""
    from whestbench.sdk import BaseEstimator

    class ZerosEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            return we.zeros((mlp.depth, mlp.width), dtype=we.float32)

    spec = ContestSpec(
        width=8, depth=2, n_mlps=1, flop_budget=100_000_000, ground_truth_samples=200
    )
    data = make_contest(spec)
    result = evaluate_estimator(ZerosEstimator(), data)
    assert "flops_used" in result["per_mlp"][0]


def test_evaluate_estimator_catches_wall_time_exhaustion() -> None:
    """When wall_time_limit_s is set and exceeded, predictions are zeroed with time_exhausted=True."""
    from whestbench.sdk import BaseEstimator

    class SlowEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            arr = we.zeros((mlp.depth, mlp.width), dtype=we.float32)
            time.sleep(0.3)
            return arr

    spec = ContestSpec(
        width=8,
        depth=2,
        n_mlps=1,
        flop_budget=100_000_000,
        ground_truth_samples=200,
        wall_time_limit_s=0.1,
    )
    data = make_contest(spec)
    result = evaluate_estimator(SlowEstimator(), data)
    mlp_result = result["per_mlp"][0]
    assert mlp_result.get("time_exhausted") is True
    assert mlp_result.get("budget_exhausted") is not True


def test_evaluate_estimator_catches_untracked_time_exhaustion() -> None:
    """When untracked_time_limit_s is exceeded, predictions are zeroed with untracked_time_exhausted=True."""
    from whestbench.sdk import BaseEstimator

    class UntrackedTimeEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            time.sleep(0.3)
            return we.zeros((mlp.depth, mlp.width), dtype=we.float32)

    spec = ContestSpec(
        width=8,
        depth=2,
        n_mlps=1,
        flop_budget=100_000_000,
        ground_truth_samples=200,
        untracked_time_limit_s=0.1,
    )
    data = make_contest(spec)
    result = evaluate_estimator(UntrackedTimeEstimator(), data)
    mlp_result = result["per_mlp"][0]
    assert mlp_result.get("untracked_time_exhausted") is True
    assert mlp_result.get("budget_exhausted") is not True


def test_evaluate_estimator_reports_timing() -> None:
    """Per-MLP results include wall_time_s, tracked_time_s, untracked_time_s."""
    from whestbench.sdk import BaseEstimator

    class ZerosEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            return we.zeros((mlp.depth, mlp.width), dtype=we.float32)

    spec = ContestSpec(
        width=8, depth=2, n_mlps=1, flop_budget=100_000_000, ground_truth_samples=200
    )
    data = make_contest(spec)
    result = evaluate_estimator(ZerosEstimator(), data)
    mlp_result = result["per_mlp"][0]
    assert "wall_time_s" in mlp_result
    assert "tracked_time_s" in mlp_result
    assert "untracked_time_s" in mlp_result
    assert mlp_result["wall_time_s"] >= 0.0
    assert mlp_result["tracked_time_s"] >= 0.0
    assert mlp_result["untracked_time_s"] >= 0.0
    assert mlp_result.get("time_exhausted") is False
    assert mlp_result.get("untracked_time_exhausted") is False
