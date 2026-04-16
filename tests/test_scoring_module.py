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


def test_make_contest_records_sampling_budget_breakdown() -> None:
    spec = ContestSpec(width=8, depth=2, n_mlps=2, flop_budget=1_000_000, ground_truth_samples=50)
    data = make_contest(spec)
    assert data.sampling_budget_breakdown is not None
    breakdown = data.sampling_budget_breakdown
    assert breakdown["flops_used"] > 0
    assert "sampling.sample_layer_statistics" in breakdown["by_namespace"]


def test_make_contest_progress_callback_preserves_sampling_budget_breakdown() -> None:
    seen: list[int] = []
    spec = ContestSpec(width=8, depth=2, n_mlps=3, flop_budget=1_000_000, ground_truth_samples=50)
    data = make_contest(spec, on_mlp_done=seen.append)
    assert seen == [1, 2, 3]
    assert data.sampling_budget_breakdown is not None
    assert data.sampling_budget_breakdown["flops_used"] > 0
    assert "sampling.sample_layer_statistics" in data.sampling_budget_breakdown["by_namespace"]


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


def test_evaluate_estimator_normalizes_explicit_namespaces() -> None:
    from whestbench.sdk import BaseEstimator

    class NamespacedEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            base = we.zeros((mlp.depth, mlp.width), dtype=we.float32)
            with we.namespace("phase"):
                return base + 1.0

    spec = ContestSpec(
        width=8, depth=2, n_mlps=1, flop_budget=100_000_000, ground_truth_samples=200
    )
    data = make_contest(spec)
    result = evaluate_estimator(NamespacedEstimator(), data)
    assert "budget_breakdown" not in result
    assert "sampling" in result["breakdowns"]
    assert "estimator" in result["breakdowns"]
    mlp_result = result["per_mlp"][0]
    breakdown = mlp_result["breakdowns"]["estimator"]
    assert breakdown is not None
    assert mlp_result["flops_used"] > 0
    assert breakdown["flops_used"] == mlp_result["flops_used"]
    assert "estimator.phase" in breakdown["by_namespace"]
    assert breakdown["by_namespace"]["estimator.phase"]["flops_used"] > 0


def test_evaluate_estimator_synthesizes_estimator_client_bucket() -> None:
    from whestbench.sdk import BaseEstimator

    class UnlabeledEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            return we.zeros((mlp.depth, mlp.width), dtype=we.float32) + 1.0

    spec = ContestSpec(
        width=8, depth=2, n_mlps=1, flop_budget=100_000_000, ground_truth_samples=200
    )
    data = make_contest(spec)
    result = evaluate_estimator(UnlabeledEstimator(), data)
    breakdown = result["per_mlp"][0]["breakdowns"]["estimator"]
    assert breakdown is not None
    assert "estimator.estimator-client" in breakdown["by_namespace"]
    assert breakdown["by_namespace"]["estimator.estimator-client"]["flops_used"] > 0


def test_evaluate_estimator_aggregates_breakdowns_across_mlps() -> None:
    from whestbench.sdk import BaseEstimator

    class NamespacedEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            base = we.zeros((mlp.depth, mlp.width), dtype=we.float32)
            with we.namespace("phase"):
                return base + 1.0

    spec = ContestSpec(
        width=8, depth=2, n_mlps=2, flop_budget=100_000_000, ground_truth_samples=200
    )
    data = make_contest(spec)
    result = evaluate_estimator(NamespacedEstimator(), data)
    assert "budget_breakdown" not in result
    estimator = result["breakdowns"]["estimator"]
    sampling = result["breakdowns"]["sampling"]
    assert estimator["by_namespace"]["estimator.phase"]["flops_used"] > 0
    assert estimator["by_namespace"]["estimator.phase"]["calls"] >= 2
    assert sampling["flops_used"] > 0


def test_evaluate_estimator_stores_per_mlp_estimator_breakdown_under_breakdowns_key() -> None:
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
    assert "budget_breakdown" not in mlp_result
    assert "breakdowns" in mlp_result
    assert "estimator" in mlp_result["breakdowns"]


def test_evaluate_estimator_prefers_runner_metadata_when_available() -> None:
    from whestbench.sdk import BaseEstimator

    class MetadataEstimator(BaseEstimator):
        def __init__(self):
            self._stats = {
                "flops_used": 123,
                "wall_time_s": 0.4,
                "tracked_time_s": 0.3,
                "untracked_time_s": 0.1,
                "budget_breakdown": {
                    "flop_budget": 1000,
                    "flops_used": 123,
                    "flops_remaining": 877,
                    "wall_time_s": 0.4,
                    "tracked_time_s": 0.3,
                    "untracked_time_s": 0.1,
                    "by_namespace": {
                        "phase": {
                            "flops_used": 123,
                            "calls": 1,
                            "tracked_time_s": 0.3,
                            "operations": {
                                "add": {"flop_cost": 123, "calls": 1, "duration": 0.3}
                            },
                        }
                    },
                },
            }

        def predict(self, mlp, budget):
            return we.zeros((mlp.depth, mlp.width), dtype=we.float32)

        def last_predict_stats(self):
            return self._stats

    spec = ContestSpec(
        width=8, depth=2, n_mlps=1, flop_budget=100_000_000, ground_truth_samples=200
    )
    data = make_contest(spec)
    result = evaluate_estimator(MetadataEstimator(), data)
    mlp_result = result["per_mlp"][0]
    assert mlp_result["flops_used"] == 123
    assert mlp_result["breakdowns"]["estimator"]["by_namespace"]["estimator.phase"]["flops_used"] == 123


def test_evaluate_estimator_synthesizes_estimator_client_for_empty_namespace_breakdown() -> None:
    from whestbench.sdk import BaseEstimator

    class MetadataEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            return we.zeros((mlp.depth, mlp.width), dtype=we.float32)

        def last_predict_stats(self):
            return {
                "flops_used": 17,
                "wall_time_s": 0.2,
                "tracked_time_s": 0.1,
                "untracked_time_s": 0.1,
                "budget_breakdown": {
                    "flop_budget": 100,
                    "flops_used": 17,
                    "flops_remaining": 83,
                    "wall_time_s": 0.2,
                    "tracked_time_s": 0.1,
                    "untracked_time_s": 0.1,
                    "by_namespace": {},
                    "operations": {},
                },
            }

    spec = ContestSpec(
        width=8, depth=2, n_mlps=1, flop_budget=100_000_000, ground_truth_samples=200
    )
    data = make_contest(spec)
    result = evaluate_estimator(MetadataEstimator(), data)
    breakdown = result["per_mlp"][0]["breakdowns"]["estimator"]
    assert breakdown is not None
    assert breakdown["by_namespace"]["estimator.estimator-client"]["flops_used"] == 17


def test_evaluate_estimator_merges_colliding_normalized_namespaces() -> None:
    from whestbench.sdk import BaseEstimator

    class MetadataEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            return we.zeros((mlp.depth, mlp.width), dtype=we.float32)

        def last_predict_stats(self):
            return {
                "flops_used": 30,
                "wall_time_s": 0.2,
                "tracked_time_s": 0.03,
                "untracked_time_s": 0.17,
                "budget_breakdown": {
                    "flop_budget": 100,
                    "flops_used": 30,
                    "flops_remaining": 70,
                    "wall_time_s": 0.2,
                    "tracked_time_s": 0.03,
                    "untracked_time_s": 0.17,
                    "by_namespace": {
                        None: {
                            "flops_used": 10,
                            "calls": 1,
                            "tracked_time_s": 0.01,
                            "operations": {
                                "add": {"flop_cost": 10, "calls": 1, "duration": 0.01}
                            },
                        },
                        "estimator-client": {
                            "flops_used": 20,
                            "calls": 2,
                            "tracked_time_s": 0.02,
                            "operations": {
                                "mul": {"flop_cost": 20, "calls": 2, "duration": 0.02}
                            },
                        },
                    },
                },
            }

    spec = ContestSpec(
        width=8, depth=2, n_mlps=1, flop_budget=100_000_000, ground_truth_samples=200
    )
    data = make_contest(spec)
    result = evaluate_estimator(MetadataEstimator(), data)
    breakdown = result["per_mlp"][0]["breakdowns"]["estimator"]
    assert breakdown is not None
    bucket = breakdown["by_namespace"]["estimator.estimator-client"]
    assert bucket["flops_used"] == 30
    assert bucket["calls"] == 3
    assert bucket["tracked_time_s"] == pytest.approx(0.03)
    assert bucket["operations"]["add"]["flop_cost"] == 10
    assert bucket["operations"]["mul"]["flop_cost"] == 20


def test_evaluate_estimator_preserves_partial_breakdown_on_budget_exhaustion() -> None:
    from whestbench.sdk import BaseEstimator

    class ExhaustingEstimator(BaseEstimator):
        def predict(self, mlp, budget):
            acc = we.zeros((mlp.depth, mlp.width), dtype=we.float32)
            with we.namespace("phase"):
                for _ in range(20):
                    acc = acc + 1.0
            return acc

    spec = ContestSpec(width=8, depth=2, n_mlps=1, flop_budget=50, ground_truth_samples=200)
    data = make_contest(spec)
    result = evaluate_estimator(ExhaustingEstimator(), data)
    mlp_result = result["per_mlp"][0]
    assert mlp_result["budget_exhausted"] is True
    assert mlp_result["breakdowns"]["estimator"]["by_namespace"]["estimator.phase"]["flops_used"] > 0
