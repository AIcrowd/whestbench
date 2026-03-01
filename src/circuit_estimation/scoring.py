"""Scoring loop, baseline timing, and optional profiling diagnostics."""

from __future__ import annotations

import platform
import socket
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit
from .generation import random_circuit
from .simulation import empirical_mean, run_batched

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX environments
    resource = None

try:
    import psutil  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

EstimatorFn = Callable[[Circuit, int], NDArray[np.float32]]
ProfilerFn = Callable[[dict[str, float | int]], None]
T = TypeVar("T")


@dataclass(slots=True)
class ContestParams:
    """Evaluator configuration shared across one scoring run.

    Attributes:
        width: Wire count for generated/scored circuits.
        max_depth: Number of layers each estimator call must predict.
        budgets: Runtime budget points used for score aggregation.
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


def score_estimator_report(
    estimator: EstimatorFn,
    n_circuits: int,
    n_samples: int,
    contest_params: ContestParams = default_contest_params,
    *,
    circuits: Sequence[Circuit] | None = None,
    profile: bool = False,
    detail: str = "raw",
    profiler: ProfilerFn | None = None,
) -> dict[str, Any]:
    """Compute a structured scoring report for one estimator.

    The report includes:
    - run metadata and configuration,
    - per-budget per-layer accuracy/runtime series,
    - optional call-level profiling events,
    - optional derived aggregates (``detail='full'``).

    Runtime handling per estimator call:
    - timeout above upper tolerance -> output tensor zeroed,
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

    n_circuits_effective = len(circuits_to_score)
    circuits_meta: list[dict[str, int]] = [
        {
            "circuit_index": idx,
            "wire_count": c.n,
            "layer_count": c.d,
        }
        for idx, c in enumerate(circuits_to_score)
    ]
    means: NDArray[np.float32] = np.array(
        [list(empirical_mean(circuit, n_samples)) for circuit in circuits_to_score],
        dtype=np.float32,
    )
    by_budget_raw: list[dict[str, Any]] = []
    profile_calls: list[dict[str, float | int]] = []
    for budget in contest_params.budgets:
        baseline_times = np.array(sampling_baseline_time(budget, width, depth), dtype=np.float32)
        baseline_times = np.maximum(baseline_times, np.float32(1e-9))
        runtimes = np.zeros(depth, dtype=np.float32)
        all_outputs: list[list[NDArray[np.float32]]] = []
        baseline_total_time = float(np.sum(baseline_times))
        timeout_counts = np.zeros(depth, dtype=np.float32)
        floor_counts = np.zeros(depth, dtype=np.float32)

        for circuit_index, circuit in enumerate(circuits_to_score):
            start_wall = time.time()
            start_cpu = time.process_time()
            raw_outputs = estimator(circuit, budget)
            elapsed = time.time() - start_wall
            cpu_elapsed = time.process_time() - start_cpu
            rss_bytes = _rss_bytes()
            peak_rss_bytes = max(_peak_rss_bytes(), rss_bytes)

            event = {
                "budget": int(budget),
                "circuit_index": int(circuit_index),
                "wire_count": int(width),
                "layer_count": int(depth),
                "wall_time_s": float(elapsed),
                "cpu_time_s": float(cpu_elapsed),
                "rss_bytes": int(rss_bytes),
                "peak_rss_bytes": int(peak_rss_bytes),
            }
            if collect_profile:
                profile_calls.append(event)
            if profiler is not None:
                profiler(event)

            if not isinstance(raw_outputs, np.ndarray):
                raise ValueError(
                    "Estimator must return a numpy.ndarray of shape (max_depth, width)."
                )

            output_tensor = np.asarray(raw_outputs, dtype=np.float32)
            if output_tensor.ndim != 2:
                raise ValueError(
                    f"Estimator output must be rank-2 with shape ({depth}, {width}), got {output_tensor.shape}."
                )
            if output_tensor.shape[0] != depth:
                raise ValueError(
                    f"Estimator yielded {output_tensor.shape[0]} outputs but expected {depth} (max_depth)."
                )
            if output_tensor.shape[1] != width:
                raise ValueError(
                    f"Estimator output width mismatch: expected output width {width}, got {output_tensor.shape}."
                )

            timed_out = elapsed > baseline_total_time * (1.0 + tolerance)
            floored = elapsed < (1.0 - tolerance) * baseline_total_time
            effective_total_time = max(elapsed, (1.0 - tolerance) * baseline_total_time)
            effective_time_per_depth = np.float32(effective_total_time / depth)

            if timed_out:
                output_tensor = np.zeros_like(output_tensor)
                timeout_counts += np.float32(1.0)
            if floored:
                floor_counts += np.float32(1.0)

            runtimes += effective_time_per_depth
            all_outputs.append([output_tensor[i] for i in range(depth)])

        estimates = np.array(all_outputs, dtype=np.float32)
        average_times = runtimes / np.float32(n_circuits_effective)
        time_ratios = average_times / baseline_times
        mse = ((estimates - means) ** 2).mean(axis=(0, 2))
        adjusted_mse = mse * time_ratios

        score = float(np.mean(adjusted_mse))
        by_budget_raw.append(
            {
                "budget": int(budget),
                "score": score,
                "mse_by_layer": mse.astype(np.float64).tolist(),
                "time_ratio_by_layer": time_ratios.astype(np.float64).tolist(),
                "adjusted_mse_by_layer": adjusted_mse.astype(np.float64).tolist(),
                "timeout_flag_by_layer": (timeout_counts / np.float32(n_circuits_effective))
                .astype(np.float64)
                .tolist(),
                "time_floor_flag_by_layer": (floor_counts / np.float32(n_circuits_effective))
                .astype(np.float64)
                .tolist(),
                "baseline_time_s_by_layer": baseline_times.astype(np.float64).tolist(),
                "effective_time_s_by_layer": average_times.astype(np.float64).tolist(),
            }
        )

    final_score = float(np.mean([entry["score"] for entry in by_budget_raw]))
    run_end = datetime.now(timezone.utc)
    host_meta = {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
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
    scores = [float(entry["score"]) for entry in by_budget_raw]
    mse_matrix = np.array([entry["mse_by_layer"] for entry in by_budget_raw], dtype=np.float64)
    adjusted_mse_matrix = np.array(
        [entry["adjusted_mse_by_layer"] for entry in by_budget_raw],
        dtype=np.float64,
    )
    time_ratio_matrix = np.array(
        [entry["time_ratio_by_layer"] for entry in by_budget_raw],
        dtype=np.float64,
    )
    timeout_matrix = np.array(
        [entry["timeout_flag_by_layer"] for entry in by_budget_raw],
        dtype=np.float64,
    )
    floor_matrix = np.array(
        [entry["time_floor_flag_by_layer"] for entry in by_budget_raw],
        dtype=np.float64,
    )
    baseline_time_matrix = np.array(
        [entry["baseline_time_s_by_layer"] for entry in by_budget_raw],
        dtype=np.float64,
    )
    effective_time_matrix = np.array(
        [entry["effective_time_s_by_layer"] for entry in by_budget_raw],
        dtype=np.float64,
    )

    by_budget_summary = [
        {
            "budget": int(budget),
            "score": float(score),
            "mse_mean": float(np.mean(mse)),
            "adjusted_mse_mean": float(np.mean(adjusted)),
            "time_ratio_mean": float(np.mean(ratio)),
            "baseline_time_s_mean": float(np.mean(baseline)),
            "effective_time_s_mean": float(np.mean(effective)),
            "timeout_rate_mean": float(np.mean(timeout)),
            "time_floor_rate_mean": float(np.mean(floor)),
        }
        for budget, score, mse, adjusted, ratio, baseline, effective, timeout, floor in zip(
            budgets,
            scores,
            mse_matrix,
            adjusted_mse_matrix,
            time_ratio_matrix,
            baseline_time_matrix,
            effective_time_matrix,
            timeout_matrix,
            floor_matrix,
            strict=True,
        )
    ]

    by_layer_overall = {
        "layer_index": list(range(depth)),
        "mse_mean_by_layer": np.mean(mse_matrix, axis=0).astype(np.float64).tolist(),
        "adjusted_mse_mean_by_layer": np.mean(adjusted_mse_matrix, axis=0)
        .astype(np.float64)
        .tolist(),
        "time_ratio_mean_by_layer": np.mean(time_ratio_matrix, axis=0).astype(np.float64).tolist(),
        "baseline_time_s_mean_by_layer": np.mean(baseline_time_matrix, axis=0)
        .astype(np.float64)
        .tolist(),
        "effective_time_s_mean_by_layer": np.mean(effective_time_matrix, axis=0)
        .astype(np.float64)
        .tolist(),
        "timeout_rate_mean_by_layer": np.mean(timeout_matrix, axis=0).astype(np.float64).tolist(),
        "time_floor_rate_mean_by_layer": np.mean(floor_matrix, axis=0).astype(np.float64).tolist(),
    }

    by_budget_layer_matrix = {
        "budgets": budgets,
        "mse_by_budget_layer": mse_matrix.astype(np.float64).tolist(),
        "adjusted_mse_by_budget_layer": adjusted_mse_matrix.astype(np.float64).tolist(),
        "time_ratio_by_budget_layer": time_ratio_matrix.astype(np.float64).tolist(),
        "baseline_time_s_by_budget_layer": baseline_time_matrix.astype(np.float64).tolist(),
        "effective_time_s_by_budget_layer": effective_time_matrix.astype(np.float64).tolist(),
        "timeout_rate_by_budget_layer": timeout_matrix.astype(np.float64).tolist(),
        "time_floor_rate_by_budget_layer": floor_matrix.astype(np.float64).tolist(),
    }

    return {
        "by_budget_summary": by_budget_summary,
        "by_layer_overall": by_layer_overall,
        "by_budget_layer_matrix": by_budget_layer_matrix,
    }


def _profile_summary(profile_calls: list[dict[str, float | int]]) -> dict[str, Any]:
    """Summarize profiling events with basic distribution statistics."""
    if not profile_calls:
        return {"call_count": 0}
    wall = np.array([float(event["wall_time_s"]) for event in profile_calls], dtype=np.float64)
    cpu = np.array([float(event["cpu_time_s"]) for event in profile_calls], dtype=np.float64)
    rss = np.array([float(event["rss_bytes"]) for event in profile_calls], dtype=np.float64)
    peak = np.array([float(event["peak_rss_bytes"]) for event in profile_calls], dtype=np.float64)
    return {
        "call_count": int(len(profile_calls)),
        "wall_time_s": {
            "mean": float(np.mean(wall)),
            "min": float(np.min(wall)),
            "max": float(np.max(wall)),
            "p95": float(np.percentile(wall, 95)),
        },
        "cpu_time_s": {
            "mean": float(np.mean(cpu)),
            "min": float(np.min(cpu)),
            "max": float(np.max(cpu)),
            "p95": float(np.percentile(cpu, 95)),
        },
        "rss_bytes": {
            "mean": float(np.mean(rss)),
            "min": float(np.min(rss)),
            "max": float(np.max(rss)),
            "p95": float(np.percentile(rss, 95)),
        },
        "peak_rss_bytes": {
            "mean": float(np.mean(peak)),
            "min": float(np.min(peak)),
            "max": float(np.max(peak)),
            "p95": float(np.percentile(peak, 95)),
        },
    }
