"""All failure modes converge to s_m = MSE(0, Y) * 1.0 with no compute discount."""

from __future__ import annotations

import flopscope.numpy as fnp
import pytest

from whestbench.domain import MLP
from whestbench.runner import RunnerError, RunnerErrorDetail
from whestbench.scoring import ContestData, ContestSpec, evaluate_estimator
from whestbench.sdk import BaseEstimator


def _make_data(width: int = 4, depth: int = 2, n_mlps: int = 1) -> ContestData:
    def weights_per_mlp() -> list:
        return [fnp.array(fnp.zeros((width, width), dtype=fnp.float32)) for _ in range(depth)]

    mlps = [MLP(width=width, depth=depth, weights=weights_per_mlp()) for _ in range(n_mlps)]
    final_target = fnp.array([1.0, 2.0, 3.0, 4.0], dtype=fnp.float32)
    all_target = fnp.array([[0.5, 1.0, 1.5, 2.0], [1.0, 2.0, 3.0, 4.0]], dtype=fnp.float32)
    return ContestData(
        spec=ContestSpec(
            width=width,
            depth=depth,
            n_mlps=n_mlps,
            flop_budget=10_000_000_000,
            ground_truth_samples=100,
        ),
        mlps=mlps,
        all_layer_targets=[all_target] * n_mlps,
        final_targets=[final_target] * n_mlps,
        avg_variances=[0.0] * n_mlps,
    )


# Expected MSE(0, [1,2,3,4]) = (1+4+9+16)/4 = 7.5.
EXPECTED_ZERO_PRED_MSE = 7.5


class _ExceptionEstimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        raise RuntimeError("estimator boom")


class _WrongShapeEstimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return fnp.zeros((mlp.width, mlp.depth))  # Transposed


class _NonFiniteEstimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        out = fnp.ones((mlp.depth, mlp.width))
        return out * fnp.array(float("inf"))


class _RunnerErrorEstimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        raise RunnerError(
            "predict",
            RunnerErrorDetail(code="PREDICT_TIMEOUT", message="predict timed out."),
        )


def test_generic_exception_routes_to_zero_pred_mse():
    """An estimator that raises gets s_m = MSE(0, Y) * 1.0, NOT inf."""
    data = _make_data()
    result = evaluate_estimator(_ExceptionEstimator(), data)
    per_mlp = result["per_mlp"][0]
    assert per_mlp.get("error_code") == "RuntimeError"
    assert per_mlp["budget_adjusted_score"] == pytest.approx(EXPECTED_ZERO_PRED_MSE, abs=1e-5)
    assert result["primary_score"] == pytest.approx(EXPECTED_ZERO_PRED_MSE, abs=1e-5)
    # Critical: suite mean must be finite, not inf.
    assert result["primary_score"] != float("inf")


def test_invalid_shape_routes_to_zero_pred_mse():
    """Returning a transposed array still scores against zero with multiplier=1.0."""
    data = _make_data()
    result = evaluate_estimator(_WrongShapeEstimator(), data)
    per_mlp = result["per_mlp"][0]
    assert per_mlp["budget_adjusted_score"] == pytest.approx(EXPECTED_ZERO_PRED_MSE, abs=1e-5)
    assert result["primary_score"] != float("inf")


def test_non_finite_values_route_to_zero_pred_mse():
    """Returning inf/NaN values scores against zero with multiplier=1.0."""
    data = _make_data()
    result = evaluate_estimator(_NonFiniteEstimator(), data)
    per_mlp = result["per_mlp"][0]
    assert per_mlp["budget_adjusted_score"] == pytest.approx(EXPECTED_ZERO_PRED_MSE, abs=1e-5)
    assert result["primary_score"] != float("inf")


def test_runner_error_routes_to_zero_pred_mse():
    """Subprocess RunnerError (timeout, OOM-killed, etc.) scores against zero."""
    data = _make_data()
    result = evaluate_estimator(_RunnerErrorEstimator(), data)
    per_mlp = result["per_mlp"][0]
    assert per_mlp["budget_adjusted_score"] == pytest.approx(EXPECTED_ZERO_PRED_MSE, abs=1e-5)
    assert result["primary_score"] != float("inf")


def test_one_failure_does_not_propagate_inf_to_suite_mean():
    """With one good MLP and one failing MLP, suite mean is finite."""
    data = _make_data(n_mlps=2)

    call_count = [0]

    class _MixedEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
            call_count[0] += 1
            if call_count[0] == 1:
                return fnp.zeros((mlp.depth, mlp.width))  # Succeeds; trivial zero
            raise RuntimeError("second MLP fails")

    result = evaluate_estimator(_MixedEstimator(), data)
    assert result["primary_score"] != float("inf")
    assert result["per_mlp"][0]["budget_adjusted_score"] == pytest.approx(
        EXPECTED_ZERO_PRED_MSE * 0.5, abs=1e-5
    )
    assert result["per_mlp"][1]["budget_adjusted_score"] == pytest.approx(
        EXPECTED_ZERO_PRED_MSE * 1.0, abs=1e-5
    )
    assert result["primary_score"] == pytest.approx(
        (EXPECTED_ZERO_PRED_MSE * 0.5 + EXPECTED_ZERO_PRED_MSE * 1.0) / 2.0, abs=1e-5
    )
