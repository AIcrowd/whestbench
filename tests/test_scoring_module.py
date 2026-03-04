from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from circuit_estimation.domain import Circuit, Layer
from circuit_estimation.runner import (
    DepthRowOutcome,
    EstimatorEntrypoint,
    ResourceLimits,
)
from circuit_estimation.scoring import (
    ContestParams,
    score_estimator,
    score_estimator_report,
)


def _constant_circuit(n: int, d: int, value: float = 1.0) -> Circuit:
    layers = []
    for _ in range(d):
        layers.append(
            Layer(
                first=np.array([0] * n, dtype=np.int32),
                second=np.array([1] * n, dtype=np.int32),
                first_coeff=np.zeros(n, dtype=np.float32),
                second_coeff=np.zeros(n, dtype=np.float32),
                const=np.array([value] * n, dtype=np.float32),
                product_coeff=np.zeros(n, dtype=np.float32),
            )
        )
    return Circuit(n=n, d=d, gates=layers)


def test_score_estimator_wrong_output_width_produces_error_row() -> None:
    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(n=2, d=1, value=1.0)

    def bad_estimator(_circuit: Circuit, _budget: int):
        yield np.array([0.0], dtype=np.float32)

    # Wrong shape is caught by validate_depth_row in _predict_depth_rows_from_fn,
    # producing a DepthRowOutcome(status='error') which is zero-filled.
    report = score_estimator_report(
        bad_estimator,
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=[circuit],
    )
    # Zero-filled row has positive MSE
    assert report["results"]["final_score"] >= 0.0


def test_score_estimator_non_iterable_output_produces_error_row() -> None:
    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(n=2, d=1, value=1.0)

    def bad_estimator(_circuit: Circuit, _budget: int):
        return 123.0

    report = score_estimator_report(
        cast(Any, bad_estimator),
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=[circuit],
    )
    assert report["results"]["final_score"] >= 0.0


def test_score_estimator_profile_hook_collects_runtime_cpu_and_memory() -> None:
    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(n=2, d=1, value=1.0)
    collected: list[dict[str, float | int]] = []

    def profiler(event: dict[str, float | int]) -> None:
        collected.append(event)

    def estimator(_circuit: Circuit, _budget: int):
        yield np.array([1.0, 1.0], dtype=np.float32)

    score_estimator(
        estimator,
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=[circuit],
        profiler=profiler,
    )

    assert collected
    last = collected[-1]
    assert {
        "budget",
        "circuit_index",
        "wire_count",
        "layer_count",
        "wall_time_s",
    } <= set(last.keys())
    assert "depth_index" not in last


def test_score_estimator_early_stop_produces_error_row() -> None:
    params = ContestParams(width=2, max_depth=2, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(n=2, d=2, value=1.0)

    def short_estimator(_circuit: Circuit, _budget: int):
        yield np.array([1.0, 1.0], dtype=np.float32)

    report = score_estimator_report(
        short_estimator,
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=[circuit],
    )
    assert report["results"]["final_score"] >= 0.0


def test_score_estimator_extra_rows_are_silently_consumed() -> None:
    params = ContestParams(width=2, max_depth=2, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(n=2, d=2, value=1.0)

    def long_estimator(_circuit: Circuit, _budget: int):
        yield np.array([1.0, 1.0], dtype=np.float32)
        yield np.array([1.0, 1.0], dtype=np.float32)
        yield np.array([1.0, 1.0], dtype=np.float32)

    # Extra rows are silently consumed
    score = score_estimator(
        long_estimator,
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=[circuit],
    )
    assert np.isfinite(score)


def test_generator_exception_after_partial_rows_produces_error_outcome() -> None:
    params = ContestParams(width=2, max_depth=2, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(n=2, d=2, value=1.0)

    def unstable_estimator(_circuit: Circuit, _budget: int):
        yield np.array([1.0, 1.0], dtype=np.float32)
        raise RuntimeError("boom")

    report = score_estimator_report(
        unstable_estimator,
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=[circuit],
    )
    # Error produces zero-filled remaining rows
    assert report["results"]["final_score"] >= 0.0


def test_non_iterable_predict_output_produces_error_row() -> None:
    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(n=2, d=1, value=1.0)

    def bad_estimator(_circuit: Circuit, _budget: int):
        return 123.0

    report = score_estimator_report(
        cast(Any, bad_estimator),
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=[circuit],
    )
    assert report["results"]["final_score"] >= 0.0


def test_row_dtype_cast_and_finite_validation_are_enforced() -> None:
    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(n=2, d=1, value=1.0)

    def int_row_estimator(_circuit: Circuit, _budget: int):
        yield np.array([1, 1], dtype=np.int32)

    score = score_estimator(
        cast(Any, int_row_estimator),
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=[circuit],
    )
    assert np.isfinite(score)

    def non_finite_estimator(_circuit: Circuit, _budget: int):
        yield np.array([np.nan, 1.0], dtype=np.float32)

    # NaN validation error becomes error outcome -> zero-filled row -> finite score
    report = score_estimator_report(
        non_finite_estimator,
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=[circuit],
    )
    assert report["results"]["final_score"] >= 0.0


def test_score_estimator_report_collects_one_profile_call_per_circuit_budget() -> None:
    params = ContestParams(width=2, max_depth=2, budgets=[10, 100], time_tolerance=0.1)
    circuits = [_constant_circuit(n=2, d=2, value=1.0) for _ in range(3)]

    def estimator(_circuit: Circuit, _budget: int):
        for _ in range(2):
            yield np.ones((2,), dtype=np.float32)

    report = score_estimator_report(
        estimator,
        n_circuits=3,
        n_samples=4,
        contest_params=params,
        circuits=circuits,
        profile=True,
    )

    profile_calls = report["profile_calls"]
    assert len(profile_calls) == len(circuits) * len(params.budgets)
    sample = profile_calls[0]
    assert {
        "budget",
        "circuit_index",
        "wire_count",
        "layer_count",
        "wall_time_s",
    } <= set(sample.keys())


def test_score_estimator_report_emits_progress_events() -> None:
    params = ContestParams(width=2, max_depth=2, budgets=[10, 100], time_tolerance=0.1)
    circuits = [_constant_circuit(n=2, d=2, value=1.0) for _ in range(3)]
    events: list[dict[str, int]] = []

    def estimator(_circuit: Circuit, _budget: int):
        for _ in range(2):
            yield np.ones((2,), dtype=np.float32)

    report = score_estimator_report(
        estimator,
        n_circuits=3,
        n_samples=4,
        contest_params=params,
        circuits=circuits,
        progress=events.append,
    )

    assert report["results"]["final_score"] >= 0.0
    assert len(events) == len(circuits) * len(params.budgets)
    assert events[-1]["completed"] == events[-1]["total"] == len(circuits) * len(params.budgets)
    assert {"budget", "budget_index", "circuit_index", "completed", "total"} <= set(
        events[-1].keys()
    )


def test_by_budget_raw_forbids_layer_runtime_fields() -> None:
    params = ContestParams(width=2, max_depth=2, budgets=[10], time_tolerance=0.1)
    circuits = [_constant_circuit(n=2, d=2, value=1.0)]

    def estimator(_circuit: Circuit, _budget: int):
        for _ in range(2):
            yield np.ones((2,), dtype=np.float32)

    report = score_estimator_report(
        estimator,
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=circuits,
        detail="raw",
    )

    row = report["results"]["by_budget_raw"][0]
    assert "time_ratio_by_layer" not in row
    assert "adjusted_mse_by_layer" not in row
    assert "baseline_time_s_by_layer" not in row
    assert "effective_time_s_by_layer" not in row
    assert "timeout_flag_by_layer" not in row
    assert "time_floor_flag_by_layer" not in row


def test_by_budget_raw_contains_scalar_and_depth_runtime_fields() -> None:
    params = ContestParams(width=2, max_depth=2, budgets=[10], time_tolerance=0.1)
    circuits = [_constant_circuit(n=2, d=2, value=1.0)]

    def estimator(_circuit: Circuit, _budget: int):
        for _ in range(2):
            yield np.ones((2,), dtype=np.float32)

    report = score_estimator_report(
        estimator,
        n_circuits=1,
        n_samples=4,
        contest_params=params,
        circuits=circuits,
        detail="raw",
    )

    row = report["results"]["by_budget_raw"][0]
    assert "mse_by_layer" in row
    assert "mse_mean" in row
    assert "adjusted_mse" in row
    assert "time_budget_by_depth_s" in row
    assert "time_ratio_by_depth_mean" in row
    assert "effective_time_s_by_depth_mean" in row
    assert "timeout_rate_by_depth" in row
    assert "time_floor_rate_by_depth" in row
    assert "call_time_ratio_mean" in row
    assert "call_effective_time_s_mean" in row
    assert "timeout_rate" in row
    assert "time_floor_rate" in row
    assert len(row["time_budget_by_depth_s"]) == params.max_depth
    assert len(row["time_ratio_by_depth_mean"]) == params.max_depth
    assert len(row["effective_time_s_by_depth_mean"]) == params.max_depth
    assert len(row["timeout_rate_by_depth"]) == params.max_depth
    assert len(row["time_floor_rate_by_depth"]) == params.max_depth


def test_detail_full_includes_budget_and_layer_aggregates() -> None:
    params = ContestParams(width=2, max_depth=2, budgets=[10, 100], time_tolerance=0.1)
    circuits = [_constant_circuit(n=2, d=2, value=1.0) for _ in range(2)]

    def estimator(_circuit: Circuit, _budget: int):
        for _ in range(2):
            yield np.ones((2,), dtype=np.float32)

    report = score_estimator_report(
        estimator,
        n_circuits=2,
        n_samples=4,
        contest_params=params,
        circuits=circuits,
        profile=True,
        detail="full",
    )

    results = report["results"]
    assert "by_budget_summary" in results
    assert "by_layer_overall" in results
    assert "by_budget_layer_matrix" in results

    by_budget_summary = results["by_budget_summary"]
    assert len(by_budget_summary) == len(params.budgets)
    assert {
        "budget",
        "mse_mean",
        "adjusted_mse",
        "call_time_ratio_mean",
        "call_effective_time_s_mean",
        "timeout_rate",
        "time_floor_rate",
    } <= set(by_budget_summary[0].keys())

    by_layer_overall = results["by_layer_overall"]
    assert by_layer_overall["layer_index"] == [0, 1]
    assert len(by_layer_overall["mse_mean_by_layer"]) == params.max_depth
    assert "adjusted_mse_mean_by_layer" not in by_layer_overall
    assert "time_ratio_mean_by_layer" not in by_layer_overall
    assert "baseline_time_s_mean_by_layer" not in by_layer_overall
    assert "effective_time_s_mean_by_layer" not in by_layer_overall

    by_budget_layer_matrix = results["by_budget_layer_matrix"]
    assert by_budget_layer_matrix["budgets"] == params.budgets
    assert len(by_budget_layer_matrix["mse_by_budget_layer"]) == len(params.budgets)
    assert "adjusted_mse_by_budget_layer" not in by_budget_layer_matrix
    assert "time_ratio_by_budget_layer" not in by_budget_layer_matrix
    assert "baseline_time_s_by_budget_layer" not in by_budget_layer_matrix
    assert "effective_time_s_by_budget_layer" not in by_budget_layer_matrix
    assert all(
        len(row) == params.max_depth for row in by_budget_layer_matrix["mse_by_budget_layer"]
    )

    assert "profile_summary" in report
    assert report["profile_summary"]["call_count"] == len(circuits) * len(params.budgets)


class _FakeStreamingRunner:
    """Fake runner that yields pre-built DepthRowOutcome sequences."""

    def __init__(self, outcomes_by_budget: dict[int, list[list[DepthRowOutcome]]]) -> None:
        self._outcomes_by_budget = outcomes_by_budget
        self._budget_indices = {budget: 0 for budget in outcomes_by_budget}
        self.started = False

    def start(self, entrypoint, context, limits) -> None:
        self.started = True

    def predict(self, circuit: Circuit, budget: int):
        idx = self._budget_indices[budget]
        self._budget_indices[budget] = idx + 1
        yield from self._outcomes_by_budget[budget][idx]

    def close(self) -> None:
        self.started = False


def test_score_estimator_report_accepts_runner_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "circuit_estimation.scoring.sampling_baseline_time",
        lambda n_samples, width, depth: [1.0] * depth,
    )
    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)
    circuits = [_constant_circuit(n=2, d=1, value=1.0) for _ in range(2)]
    outcomes_per_circuit = [
        [DepthRowOutcome(depth_index=0, row=np.array([1.0, 1.0], dtype=np.float32), wall_time_s=1.0, status="ok")],
        [DepthRowOutcome(depth_index=0, row=np.array([1.0, 1.0], dtype=np.float32), wall_time_s=1.0, status="ok")],
    ]
    runner = _FakeStreamingRunner({10: outcomes_per_circuit})
    report = score_estimator_report(
        runner,
        n_circuits=2,
        n_samples=4,
        contest_params=params,
        entrypoint=EstimatorEntrypoint(file_path=Path("estimator.py")),
        limits=ResourceLimits(setup_timeout_s=1.0, predict_timeout_s=1.0, memory_limit_mb=256),
        circuits=circuits,
        detail="raw",
    )

    row = report["results"]["by_budget_raw"][0]
    assert row["mse_mean"] == pytest.approx(0.0)
    assert row["call_time_ratio_mean"] == pytest.approx(1.0)
    assert "time_budget_by_depth_s" in row
    assert "timeout_rate_by_depth" in row
    assert report["results"]["final_score"] == pytest.approx(0.0)
    assert runner.started is False


def test_runner_error_outcomes_produce_zero_filled_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "circuit_estimation.scoring.sampling_baseline_time",
        lambda n_samples, width, depth: [1.0] * depth,
    )
    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)
    circuits = [_constant_circuit(n=2, d=1, value=1.0) for _ in range(2)]
    outcomes_per_circuit = [
        [DepthRowOutcome(depth_index=0, row=None, wall_time_s=1.0, status="error", error_message="boom")],
        [DepthRowOutcome(depth_index=0, row=np.array([1.0, 1.0], dtype=np.float32), wall_time_s=1.0, status="ok")],
    ]
    runner = _FakeStreamingRunner({10: outcomes_per_circuit})
    report = score_estimator_report(
        runner,
        n_circuits=2,
        n_samples=4,
        contest_params=params,
        entrypoint=EstimatorEntrypoint(file_path=Path("estimator.py")),
        limits=ResourceLimits(setup_timeout_s=1.0, predict_timeout_s=1.0, memory_limit_mb=256),
        circuits=circuits,
        detail="raw",
    )

    row = report["results"]["by_budget_raw"][0]
    # First circuit errored (zeros), second was perfect  -> MSE > 0
    assert row["adjusted_mse"] > 0.0
