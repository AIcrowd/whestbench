import time
from dataclasses import replace

import pytest
import whest as we

from whestbench.domain import MLP
from whestbench.scoring import (
    ContestSpec,
    evaluate_estimator,
    make_contest,
    make_contest_from_bundle,
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


def test_make_contest_is_reproducible_with_seed() -> None:
    spec = ContestSpec(
        width=8, depth=2, n_mlps=3, flop_budget=1_000_000, ground_truth_samples=64, seed=7777
    )
    data_a = make_contest(spec)
    data_b = make_contest(spec)

    for i in range(spec.n_mlps):
        for wa, wb in zip(data_a.mlps[i].weights, data_b.mlps[i].weights):
            we.testing.assert_array_equal(wa, wb)
        we.testing.assert_array_equal(data_a.all_layer_targets[i], data_b.all_layer_targets[i])
        we.testing.assert_array_equal(data_a.final_targets[i], data_b.final_targets[i])
    assert data_a.avg_variances == data_b.avg_variances


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
                            "operations": {"add": {"flop_cost": 123, "calls": 1, "duration": 0.3}},
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
    assert (
        mlp_result["breakdowns"]["estimator"]["by_namespace"]["estimator.phase"]["flops_used"]
        == 123
    )


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
                            "operations": {"add": {"flop_cost": 10, "calls": 1, "duration": 0.01}},
                        },
                        "estimator-client": {
                            "flops_used": 20,
                            "calls": 2,
                            "tracked_time_s": 0.02,
                            "operations": {"mul": {"flop_cost": 20, "calls": 2, "duration": 0.02}},
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
    assert (
        mlp_result["breakdowns"]["estimator"]["by_namespace"]["estimator.phase"]["flops_used"] > 0
    )


# --- make_contest_from_bundle ---------------------------------------------


def _build_bundle_from_contest(n_mlps: int = 4, width: int = 8, depth: int = 2):
    """Use make_contest to create MLPs/targets, wrap them in a DatasetBundle."""
    import numpy as np

    from whestbench.dataset import DatasetBundle

    spec = ContestSpec(
        width=width, depth=depth, n_mlps=n_mlps, flop_budget=1_000_000, ground_truth_samples=100
    )
    data = make_contest(spec)
    all_layer_means = np.stack([np.asarray(t) for t in data.all_layer_targets]).astype(np.float32)
    final_means = np.stack([np.asarray(t) for t in data.final_targets]).astype(np.float32)
    return DatasetBundle(
        metadata={"width": width, "depth": depth, "n_mlps": n_mlps},
        mlps=list(data.mlps),
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=list(data.avg_variances),
        sampling_budget_breakdowns=None,
    )


def test_make_contest_from_bundle_returns_full_bundle() -> None:
    import numpy as np

    bundle = _build_bundle_from_contest(n_mlps=3)
    spec = ContestSpec(width=8, depth=2, n_mlps=3, flop_budget=1_000_000, ground_truth_samples=100)
    data = make_contest_from_bundle(spec, bundle, n_mlps=3)

    assert len(data.mlps) == 3
    assert len(data.all_layer_targets) == 3
    assert len(data.final_targets) == 3
    assert len(data.avg_variances) == 3
    assert data.sampling_budget_breakdown is None
    # MLP identity preserved: bundle MLPs are passed through, not regenerated.
    for bundled, picked in zip(bundle.mlps, data.mlps):
        assert picked is bundled
    # Targets equal bundle contents element-wise.
    for i in range(3):
        assert np.allclose(np.asarray(data.all_layer_targets[i]), bundle.all_layer_means[i])
        assert np.allclose(np.asarray(data.final_targets[i]), bundle.final_means[i])


def test_make_contest_from_bundle_subsets_first_n() -> None:
    import numpy as np

    bundle = _build_bundle_from_contest(n_mlps=5)
    spec = ContestSpec(width=8, depth=2, n_mlps=2, flop_budget=1_000_000, ground_truth_samples=100)
    data = make_contest_from_bundle(spec, bundle, n_mlps=2)

    assert len(data.mlps) == 2
    # First two, not some other slice.
    assert data.mlps[0] is bundle.mlps[0]
    assert data.mlps[1] is bundle.mlps[1]
    assert np.allclose(np.asarray(data.final_targets[0]), bundle.final_means[0])
    assert np.allclose(np.asarray(data.final_targets[1]), bundle.final_means[1])


def test_make_contest_from_bundle_restores_sampling_breakdown_for_subset() -> None:
    bundle = replace(
        _build_bundle_from_contest(n_mlps=3),
        sampling_budget_breakdowns=[
            {
                "flop_budget": 1000,
                "flops_used": 10,
                "flops_remaining": 990,
                "wall_time_s": 0.01,
                "tracked_time_s": 0.005,
                "untracked_time_s": 0.002,
                "by_namespace": {
                    "sampling.sample_layer_statistics": {
                        "flops_used": 10,
                        "calls": 1,
                        "tracked_time_s": 0.005,
                        "operations": {},
                    }
                },
            },
            {
                "flop_budget": 1000,
                "flops_used": 20,
                "flops_remaining": 980,
                "wall_time_s": 0.02,
                "tracked_time_s": 0.010,
                "untracked_time_s": 0.004,
                "by_namespace": {
                    "sampling.sample_layer_statistics": {
                        "flops_used": 20,
                        "calls": 1,
                        "tracked_time_s": 0.010,
                        "operations": {},
                    }
                },
            },
            {
                "flop_budget": 1000,
                "flops_used": 30,
                "flops_remaining": 970,
                "wall_time_s": 0.03,
                "tracked_time_s": 0.015,
                "untracked_time_s": 0.006,
                "by_namespace": {
                    "sampling.sample_layer_statistics": {
                        "flops_used": 30,
                        "calls": 1,
                        "tracked_time_s": 0.015,
                        "operations": {},
                    }
                },
            },
        ],
    )

    spec = ContestSpec(width=8, depth=2, n_mlps=2, flop_budget=1_000_000, ground_truth_samples=100)
    data = make_contest_from_bundle(spec, bundle, n_mlps=2)

    assert data.sampling_budget_breakdown is not None
    assert data.sampling_budget_breakdown["flops_used"] == 30
    assert data.sampling_budget_breakdown["tracked_time_s"] == 0.015
    assert data.sampling_budget_breakdown["untracked_time_s"] == 0.006
    assert (
        data.sampling_budget_breakdown["by_namespace"]["sampling.sample_layer_statistics"][
            "flops_used"
        ]
        == 30
    )


def test_make_contest_from_bundle_rejects_oversize() -> None:
    bundle = _build_bundle_from_contest(n_mlps=2)
    spec = ContestSpec(width=8, depth=2, n_mlps=5, flop_budget=1_000_000, ground_truth_samples=100)
    with pytest.raises(ValueError, match="exceeds bundle size"):
        make_contest_from_bundle(spec, bundle, n_mlps=5)


def test_make_contest_from_bundle_rejects_non_positive() -> None:
    bundle = _build_bundle_from_contest(n_mlps=2)
    # spec.validate() rejects n_mlps<=0 before we even get there, so build a
    # valid spec and call with n_mlps=0 directly.
    spec = ContestSpec(width=8, depth=2, n_mlps=1, flop_budget=1_000_000, ground_truth_samples=100)
    with pytest.raises(ValueError, match="n_mlps must be positive"):
        make_contest_from_bundle(spec, bundle, n_mlps=0)


def test_make_contest_from_bundle_rejects_spec_mismatch() -> None:
    bundle = _build_bundle_from_contest(n_mlps=3)
    spec = ContestSpec(width=8, depth=2, n_mlps=2, flop_budget=1_000_000, ground_truth_samples=100)
    with pytest.raises(ValueError, match="spec.n_mlps"):
        make_contest_from_bundle(spec, bundle, n_mlps=3)


# --- evaluate_estimator error propagation --------------------------------


class _RaisingEstimator:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def predict(self, mlp, budget):  # type: ignore[no-untyped-def]
        raise self._exc


def test_evaluate_estimator_captures_traceback_and_error_code() -> None:
    spec = ContestSpec(width=8, depth=2, n_mlps=2, flop_budget=1_000_000, ground_truth_samples=100)
    data = make_contest(spec)
    estimator = _RaisingEstimator(RuntimeError("boom from predict"))

    result = evaluate_estimator(estimator, data)

    assert len(result["per_mlp"]) == 2
    for entry in result["per_mlp"]:
        assert entry["error"] == "boom from predict"
        # Bare predict exceptions surface as the Python class name;
        # RunnerError would surface as its .detail.code.
        assert entry["error_code"] == "RuntimeError"
        assert isinstance(entry["traceback"], str)
        assert "boom from predict" in entry["traceback"]
    assert "RuntimeError" in entry["traceback"]
    assert result["primary_score"] == float("inf")


def test_evaluate_estimator_records_validation_error_details() -> None:
    spec = ContestSpec(
        width=4, depth=3, n_mlps=1, flop_budget=100_000_000, ground_truth_samples=100
    )
    data = make_contest(spec)
    from whestbench.sdk import BaseEstimator

    class _WrongShapeEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> we.ndarray:
            return we.ones((spec.width, spec.depth), dtype=we.float32)

    result = evaluate_estimator(_WrongShapeEstimator(), data)
    entry = result["per_mlp"][0]

    assert isinstance(entry["error"], dict)
    assert entry["error_code"] == "ValueError"
    assert "shape" in str(entry["error"].get("message", "")).lower()
    details = entry["error"]["details"]
    assert details["expected_shape"] == [spec.depth, spec.width]
    assert details["got_shape"] == [spec.width, spec.depth]


def test_evaluate_estimator_prefers_runner_traceback_when_present() -> None:
    from whestbench.runner import RunnerError, RunnerErrorDetail

    spec = ContestSpec(width=8, depth=2, n_mlps=1, flop_budget=1_000_000, ground_truth_samples=100)
    data = make_contest(spec)
    runner_err = RunnerError(
        "predict",
        RunnerErrorDetail(
            code="PREDICT_ERROR",
            message="worker crashed",
            traceback="FAKE_REMOTE_TRACEBACK",
        ),
    )
    estimator = _RaisingEstimator(runner_err)

    result = evaluate_estimator(estimator, data)

    entry = result["per_mlp"][0]
    assert entry["error_code"] == "PREDICT_ERROR"
    assert entry["traceback"] == "FAKE_REMOTE_TRACEBACK"


def test_evaluate_estimator_fail_fast_re_raises() -> None:
    spec = ContestSpec(width=8, depth=2, n_mlps=3, flop_budget=1_000_000, ground_truth_samples=100)
    data = make_contest(spec)
    estimator = _RaisingEstimator(RuntimeError("abort here"))

    with pytest.raises(RuntimeError, match="abort here"):
        evaluate_estimator(estimator, data, fail_fast=True)
