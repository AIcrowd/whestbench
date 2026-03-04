"""Scoring loop, budget-by-depth timing, and optional profiling diagnostics."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit
from .generation import random_circuit
from .hardware import collect_hardware_fingerprint
from .runner import (
    DepthRowOutcome,
    EstimatorEntrypoint,
    EstimatorRunner,
    ResourceLimits,
    RunnerError,
)
from .sdk import SetupContext
from .simulation import empirical_mean, run_batched
from .streaming import validate_depth_row

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX environments
    resource = None

try:
    import psutil  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

EstimatorFn = Callable[[Circuit, int], Iterator[NDArray[np.float32]]]
ProfilerFn = Callable[[dict[str, float | int]], None]
ProgressEvent = dict[str, int | str]
ProgressFn = Callable[[ProgressEvent], None]
T = TypeVar("T")


@dataclass(slots=True)
class ContestParams:
    """Evaluator configuration shared across one scoring run.

    Attributes:
        width: Wire count for generated/scored circuits.
        max_depth: Number of layers each estimator call must predict.
        budgets: Sampling trial-count budgets used for score aggregation.
        time_tolerance: Relative slack for timeout/floor runtime semantics.
    """

    width: int
    max_depth: int
    budgets: list[int]
    time_tolerance: float

    def validate(self) -> None:
        """Validate evaluator configuration bounds and required fields."""
        if self.width <= 0:
            raise ValueError("width must be positive.")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if not self.budgets or any(b <= 0 for b in self.budgets):
            raise ValueError("budgets must be a non-empty list of positive integers.")
        if not (0.0 <= self.time_tolerance < 1.0):
            raise ValueError("time_tolerance must be in [0.0, 1.0).")


default_contest_params = ContestParams(
    width=1000,
    max_depth=300,
    budgets=[10**i for i in range(2, 6)],
    time_tolerance=0.1,
)

default_resource_limits = ResourceLimits(
    setup_timeout_s=30.0,
    predict_timeout_s=30.0,
    memory_limit_mb=4096,
    cpu_time_limit_s=None,
)


def profile_fn(
    fn: Callable[[], Iterator[T]],
) -> Iterator[tuple[float, T]]:
    """Yield cumulative wall time and each iterator output from ``fn``."""
    start_time = time.time()
    for output in fn():
        yield time.time() - start_time, output


def _peak_rss_bytes() -> int:
    """Return best-effort process peak RSS in bytes."""
    if resource is None:
        return 0
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return usage
    return usage * 1024


def _rss_bytes() -> int:
    """Return current process RSS in bytes (or fallback estimate)."""
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)
    return _peak_rss_bytes()


def sampling_baseline_time(n_samples: int, width: int, depth: int) -> list[float]:
    """Measure per-depth baseline runtime for plain sampling forward passes."""
    circuit = random_circuit(width, depth)
    inputs = np.random.choice([-1.0, 1.0], size=(n_samples, width)).astype(np.float16)
    return [elapsed for elapsed, _ in profile_fn(lambda: run_batched(circuit, inputs))]


def _predict_depth_rows_from_fn(
    estimator: EstimatorFn, circuit: Circuit, budget: int, *, width: int, depth: int,
) -> Iterator[DepthRowOutcome]:
    """Wrap a raw EstimatorFn call into a DepthRowOutcome stream with timing."""
    start_wall = time.time()
    try:
        raw_outputs = estimator(circuit, budget)
    except Exception as exc:
        yield DepthRowOutcome(
            depth_index=0, row=None,
            wall_time_s=time.time() - start_wall,
            status="error", error_message=str(exc),
        )
        return
    try:
        output_iter = iter(raw_outputs)
    except TypeError:
        yield DepthRowOutcome(
            depth_index=0, row=None,
            wall_time_s=time.time() - start_wall,
            status="error",
            error_message="Estimator must return an iterator of depth-row outputs.",
        )
        return

    for depth_index in range(depth):
        try:
            raw_row = next(output_iter)
        except StopIteration:
            yield DepthRowOutcome(
                depth_index=depth_index, row=None,
                wall_time_s=time.time() - start_wall,
                status="error",
                error_message="Estimator must emit exactly max_depth rows.",
            )
            return
        except Exception as exc:
            yield DepthRowOutcome(
                depth_index=depth_index, row=None,
                wall_time_s=time.time() - start_wall,
                status="error",
                error_message=f"Estimator stream failed at depth {depth_index}: {exc}",
            )
            return

        elapsed = time.time() - start_wall
        try:
            row = validate_depth_row(raw_row, width=width, depth_index=depth_index)
        except ValueError as exc:
            yield DepthRowOutcome(
                depth_index=depth_index, row=None,
                wall_time_s=elapsed, status="error",
                error_message=str(exc),
            )
            return

        yield DepthRowOutcome(
            depth_index=depth_index, row=row,
            wall_time_s=elapsed, status="ok",
        )

    # Check for extra rows (silently consume)
    try:
        _extra = next(output_iter)
    except StopIteration:
        pass
    except Exception:
        pass


def score_estimator_report(
    estimator: EstimatorFn | EstimatorRunner,
    n_circuits: int,
    n_samples: int,
    contest_params: ContestParams = default_contest_params,
    *,
    entrypoint: EstimatorEntrypoint | None = None,
    limits: ResourceLimits = default_resource_limits,
    circuits: Sequence[Circuit] | None = None,
    profile: bool = False,
    detail: str = "raw",
    profiler: ProfilerFn | None = None,
    progress: ProgressFn | None = None,
    sampling_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Compute a structured scoring report for one estimator.

    The estimator may be provided as a raw callable (``EstimatorFn``)
    or as an ``EstimatorRunner`` instance. When a runner is supplied,
    the caller must also provide ``entrypoint`` and ``limits`` so the
    runner can be initialised.

    The report includes:
    - run metadata and configuration,
    - per-budget per-depth accuracy/runtime series,
    - optional call-level profiling events,
    - optional derived aggregates (``detail='full'``).

    Runtime handling per depth emission:
    - timeout above upper tolerance -> depth row zeroed,
    - runtime below lower tolerance -> effective runtime floored.
    """
    contest_params.validate()
    if n_circuits <= 0:
        raise ValueError("n_circuits must be positive.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if detail not in {"raw", "full"}:
        raise ValueError("detail must be one of: raw, full.")

    width = contest_params.width
    depth = contest_params.max_depth
    tolerance = contest_params.time_tolerance
    run_start = datetime.now(timezone.utc)
    run_start_wall = time.time()
    collect_profile = profile or profiler is not None

    # Detect whether we have a runner or a raw callable
    is_runner = hasattr(estimator, "start") and hasattr(estimator, "close")
    runner: EstimatorRunner | None = None
    estimator_fn: EstimatorFn | None = None
    if is_runner:
        runner = estimator  # type: ignore[assignment]
        if entrypoint is None:
            raise ValueError("entrypoint is required when using a runner.")
    else:
        estimator_fn = estimator  # type: ignore[assignment]

    circuits_to_score = (
        list(circuits)
        if circuits is not None
        else [random_circuit(width, depth) for _ in range(n_circuits)]
    )
    if not circuits_to_score:
        raise ValueError("At least one circuit is required for scoring.")
    if any(c.n != width for c in circuits_to_score):
        raise ValueError("All circuits must have width equal to contest_params.width.")
    if any(c.d != depth for c in circuits_to_score):
        raise ValueError("All circuits must have depth equal to contest_params.max_depth.")

    # If runner, start it
    if runner is not None:
        assert entrypoint is not None
        setup_context = SetupContext(
            width=width,
            max_depth=depth,
            budgets=tuple(int(b) for b in contest_params.budgets),
            time_tolerance=tolerance,
            api_version="1.0",
        )
        try:
            runner.start(entrypoint, setup_context, limits)
        except RunnerError:
            runner.close()
            raise

    n_circuits_effective = len(circuits_to_score)
    total_units = len(contest_params.budgets) * n_circuits_effective
    circuits_meta: list[dict[str, int]] = [
        {
            "circuit_index": idx,
            "wire_count": c.n,
            "layer_count": c.d,
        }
        for idx, c in enumerate(circuits_to_score)
    ]
    means_by_circuit: list[list[NDArray[np.float32]]] = []
    for circuit_index, circuit in enumerate(circuits_to_score):
        means_by_circuit.append(list(empirical_mean(circuit, n_samples)))
        if sampling_progress is not None:
            sampling_progress(
                {
                    "phase": "sampling",
                    "circuit_index": int(circuit_index),
                    "completed": int(circuit_index + 1),
                    "total": int(n_circuits_effective),
                }
            )
    means: NDArray[np.float32] = np.array(means_by_circuit, dtype=np.float32)
    by_budget_raw: list[dict[str, Any]] = []
    profile_calls: list[dict[str, float | int]] = []
    completed_units = 0
    try:
        for budget_index, budget in enumerate(contest_params.budgets):
            baseline_times = np.array(sampling_baseline_time(budget, width, depth), dtype=np.float32)
            baseline_times = np.maximum(baseline_times, np.float32(1e-9))
            all_outputs: list[list[NDArray[np.float32]]] = []
            effective_time_sums_by_depth = np.zeros(depth, dtype=np.float64)
            timeout_counts_by_depth = np.zeros(depth, dtype=np.float64)
            floor_counts_by_depth = np.zeros(depth, dtype=np.float64)
            error_count = 0

            for circuit_index, circuit in enumerate(circuits_to_score):
                # Get per-depth outcomes from either runner or raw fn
                if runner is not None:
                    depth_outcomes = list(runner.predict(circuit, budget))
                else:
                    assert estimator_fn is not None
                    depth_outcomes = list(
                        _predict_depth_rows_from_fn(
                            estimator_fn, circuit, budget,
                            width=width, depth=depth,
                        )
                    )

                # Process depth outcomes
                rows: list[NDArray[np.float32]] = []
                has_error = False
                last_wall_time = 0.0
                for outcome in depth_outcomes:
                    if outcome.status == "error":
                        has_error = True
                        error_count += 1
                        break
                    assert outcome.row is not None
                    d_idx = outcome.depth_index
                    elapsed = outcome.wall_time_s
                    last_wall_time = elapsed
                    baseline_time = float(baseline_times[d_idx])

                    timed_out = elapsed > baseline_time * (1.0 + tolerance)
                    floored = elapsed < baseline_time * (1.0 - tolerance)
                    effective_time = max(elapsed, baseline_time * (1.0 - tolerance))

                    row = outcome.row
                    if timed_out:
                        row = np.zeros_like(row)

                    rows.append(row)
                    effective_time_sums_by_depth[d_idx] += float(effective_time)
                    timeout_counts_by_depth[d_idx] += float(timed_out)
                    floor_counts_by_depth[d_idx] += float(floored)

                if has_error or len(rows) < depth:
                    # Fill missing rows with zeros
                    while len(rows) < depth:
                        rows.append(np.zeros(width, dtype=np.float32))

                event = {
                    "budget": int(budget),
                    "circuit_index": int(circuit_index),
                    "wire_count": int(width),
                    "layer_count": int(depth),
                    "wall_time_s": float(last_wall_time),
                }
                if collect_profile:
                    profile_calls.append(event)
                if profiler is not None:
                    profiler(event)

                output_tensor = np.stack(rows, axis=0).astype(np.float32)
                all_outputs.append([output_tensor[i] for i in range(depth)])
                completed_units += 1
                if progress is not None:
                    progress(
                        {
                            "phase": "scoring",
                            "budget_index": int(budget_index),
                            "budget": int(budget),
                            "circuit_index": int(circuit_index),
                            "completed": int(completed_units),
                            "total": int(total_units),
                        }
                    )

            estimates = np.array(all_outputs, dtype=np.float32)
            mse = ((estimates - means) ** 2).mean(axis=(0, 2))
            average_effective_times_by_depth = effective_time_sums_by_depth / n_circuits_effective
            time_ratios_by_depth = average_effective_times_by_depth / baseline_times
            adjusted_mse_by_depth = mse * time_ratios_by_depth
            timeout_rate_by_depth = timeout_counts_by_depth / n_circuits_effective
            floor_rate_by_depth = floor_counts_by_depth / n_circuits_effective

            mse_mean = float(np.mean(mse))
            call_effective_time_s_mean = float(np.mean(average_effective_times_by_depth))
            call_time_ratio_mean = float(np.mean(time_ratios_by_depth))
            adjusted_mse = float(np.mean(adjusted_mse_by_depth))
            timeout_rate = float(np.mean(timeout_rate_by_depth))
            time_floor_rate = float(np.mean(floor_rate_by_depth))
            by_budget_raw.append(
                {
                    "budget": int(budget),
                    "time_budget_by_depth_s": baseline_times.astype(np.float64).tolist(),
                    "mse_by_layer": mse.astype(np.float64).tolist(),
                    "mse_mean": mse_mean,
                    "adjusted_mse": adjusted_mse,
                    "time_ratio_by_depth_mean": time_ratios_by_depth.astype(np.float64).tolist(),
                    "effective_time_s_by_depth_mean": average_effective_times_by_depth.astype(
                        np.float64
                    ).tolist(),
                    "timeout_rate_by_depth": timeout_rate_by_depth.astype(np.float64).tolist(),
                    "time_floor_rate_by_depth": floor_rate_by_depth.astype(np.float64).tolist(),
                    "call_time_ratio_mean": call_time_ratio_mean,
                    "call_effective_time_s_mean": call_effective_time_s_mean,
                    "timeout_rate": timeout_rate,
                    "time_floor_rate": time_floor_rate,
                }
            )
    finally:
        if runner is not None:
            runner.close()

    final_score = float(np.mean([entry["adjusted_mse"] for entry in by_budget_raw]))
    run_end = datetime.now(timezone.utc)
    host_meta = collect_hardware_fingerprint()
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": "agent",
        "detail": detail,
        "run_meta": {
            "run_started_at_utc": run_start.isoformat(),
            "run_finished_at_utc": run_end.isoformat(),
            "run_duration_s": float(time.time() - run_start_wall),
            "host": host_meta,
        },
        "run_config": {
            "n_circuits": int(n_circuits_effective),
            "n_samples": int(n_samples),
            "width": int(width),
            "max_depth": int(depth),
            "layer_count": int(depth),
            "budgets": [int(b) for b in contest_params.budgets],
            "time_tolerance": float(tolerance),
            "profile_enabled": bool(collect_profile),
        },
        "circuits": circuits_meta,
        "results": {
            "final_score": final_score,
            "score_direction": "lower_is_better",
            "by_budget_raw": by_budget_raw,
        },
        "notes": [],
    }
    if collect_profile:
        report["profile_calls"] = profile_calls
        if detail == "full":
            report["profile_summary"] = _profile_summary(profile_calls)
    if detail == "full":
        report["results"].update(_compute_full_detail(by_budget_raw, depth))
    return report


def score_estimator(
    estimator: EstimatorFn,
    n_circuits: int,
    n_samples: int,
    contest_params: ContestParams = default_contest_params,
    *,
    circuits: Sequence[Circuit] | None = None,
    profiler: ProfilerFn | None = None,
) -> float:
    """Return only final score for callers that do not need full report payload."""
    report = score_estimator_report(
        estimator,
        n_circuits=n_circuits,
        n_samples=n_samples,
        contest_params=contest_params,
        circuits=circuits,
        profile=profiler is not None,
        detail="raw",
        profiler=profiler,
    )
    return float(report["results"]["final_score"])


def _compute_full_detail(by_budget_raw: list[dict[str, Any]], depth: int) -> dict[str, Any]:
    """Derive aggregate tables/matrices from raw per-budget layer series."""
    budgets = [int(entry["budget"]) for entry in by_budget_raw]
    mse_matrix = np.array([entry["mse_by_layer"] for entry in by_budget_raw], dtype=np.float64)
    mse_means = [float(entry["mse_mean"]) for entry in by_budget_raw]
    adjusted_mse = [float(entry["adjusted_mse"]) for entry in by_budget_raw]
    call_time_ratio_mean = [float(entry["call_time_ratio_mean"]) for entry in by_budget_raw]
    call_effective_time_s_mean = [
        float(entry["call_effective_time_s_mean"]) for entry in by_budget_raw
    ]
    timeout_rate = [float(entry["timeout_rate"]) for entry in by_budget_raw]
    time_floor_rate = [float(entry["time_floor_rate"]) for entry in by_budget_raw]
    runtime_error_rate = [float(entry.get("runtime_error_rate", 0.0)) for entry in by_budget_raw]
    protocol_error_rate = [float(entry.get("protocol_error_rate", 0.0)) for entry in by_budget_raw]
    oom_rate = [float(entry.get("oom_rate", 0.0)) for entry in by_budget_raw]

    by_budget_summary = [
        {
            "budget": int(budget),
            "mse_mean": mse_mean,
            "adjusted_mse": adjusted,
            "call_time_ratio_mean": ratio,
            "call_effective_time_s_mean": effective,
            "timeout_rate": timeout,
            "time_floor_rate": floor,
            "runtime_error_rate": runtime_error,
            "protocol_error_rate": protocol_error,
            "oom_rate": oom,
        }
        for budget, mse_mean, adjusted, ratio, effective, timeout, floor, runtime_error, protocol_error, oom in zip(
            budgets,
            mse_means,
            adjusted_mse,
            call_time_ratio_mean,
            call_effective_time_s_mean,
            timeout_rate,
            time_floor_rate,
            runtime_error_rate,
            protocol_error_rate,
            oom_rate,
            strict=True,
        )
    ]

    by_layer_overall = {
        "layer_index": list(range(depth)),
        "mse_mean_by_layer": np.mean(mse_matrix, axis=0).astype(np.float64).tolist(),
    }

    by_budget_layer_matrix = {
        "budgets": budgets,
        "mse_by_budget_layer": mse_matrix.astype(np.float64).tolist(),
    }

    return {
        "by_budget_summary": by_budget_summary,
        "by_layer_overall": by_layer_overall,
        "by_budget_layer_matrix": by_budget_layer_matrix,
    }


def _profile_summary(profile_calls: Sequence[Mapping[str, float | int | str]]) -> dict[str, Any]:
    """Summarize profiling events with basic distribution statistics."""
    if not profile_calls:
        return {"call_count": 0}
    wall = np.array([float(event["wall_time_s"]) for event in profile_calls], dtype=np.float64)
    return {
        "call_count": int(len(profile_calls)),
        "wall_time_s": {
            "mean": float(np.mean(wall)),
            "min": float(np.min(wall)),
            "max": float(np.max(wall)),
            "p95": float(np.percentile(wall, 95)),
        },
    }
