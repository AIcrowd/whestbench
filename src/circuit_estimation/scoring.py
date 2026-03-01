"""Scoring loop, baseline timing, and optional profiling diagnostics."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

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
    width: int
    max_depth: int
    budgets: list[int]
    time_tolerance: float

    def validate(self) -> None:
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
    if resource is None:
        return 0
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return usage
    return usage * 1024


def _rss_bytes() -> int:
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)
    return _peak_rss_bytes()


def sampling_baseline_time(n_samples: int, width: int, depth: int) -> list[float]:
    """Measure per-depth baseline runtime for plain sampling forward passes."""
    circuit = random_circuit(width, depth)
    inputs = np.random.choice([-1.0, 1.0], size=(n_samples, width)).astype(np.float16)
    return [elapsed for elapsed, _ in profile_fn(lambda: run_batched(circuit, inputs))]


def score_estimator(
    estimator: EstimatorFn,
    n_circuits: int,
    n_samples: int,
    contest_params: ContestParams = default_contest_params,
    *,
    circuits: Sequence[Circuit] | None = None,
    profiler: ProfilerFn | None = None,
) -> float:
    """Compute adjusted MSE score under runtime budgets."""
    contest_params.validate()
    if n_circuits <= 0:
        raise ValueError("n_circuits must be positive.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    width = contest_params.width
    depth = contest_params.max_depth
    tolerance = contest_params.time_tolerance

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
    means: NDArray[np.float32] = np.array(
        [list(empirical_mean(circuit, n_samples)) for circuit in circuits_to_score],
        dtype=np.float32,
    )
    variances = (1.0 - means * means).mean(axis=(0, 2))

    performance_by_budget: list[float] = []
    for budget in contest_params.budgets:
        baseline_times = np.array(sampling_baseline_time(budget, width, depth), dtype=np.float32)
        baseline_times = np.maximum(baseline_times, np.float32(1e-9))
        runtimes = np.zeros(depth, dtype=np.float32)
        all_outputs: list[list[NDArray[np.float32]]] = []
        baseline_total_time = float(np.sum(baseline_times))

        for circuit_index, circuit in enumerate(circuits_to_score):
            start_wall = time.time()
            start_cpu = time.process_time()
            raw_outputs = estimator(circuit, budget)
            elapsed = time.time() - start_wall
            cpu_elapsed = time.process_time() - start_cpu
            rss_bytes = _rss_bytes()
            peak_rss_bytes = max(_peak_rss_bytes(), rss_bytes)

            if profiler is not None:
                profiler(
                    {
                        "budget": int(budget),
                        "circuit_index": int(circuit_index),
                        "wall_time_s": float(elapsed),
                        "cpu_time_s": float(cpu_elapsed),
                        "rss_bytes": int(rss_bytes),
                        "peak_rss_bytes": int(peak_rss_bytes),
                    }
                )

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
            effective_total_time = max(elapsed, (1.0 - tolerance) * baseline_total_time)
            effective_time_per_depth = np.float32(effective_total_time / depth)

            if timed_out:
                output_tensor = np.zeros_like(output_tensor)

            runtimes += effective_time_per_depth
            all_outputs.append([output_tensor[i] for i in range(depth)])

        estimates = np.array(all_outputs, dtype=np.float32)
        average_times = runtimes / np.float32(n_circuits_effective)
        time_ratios = average_times / baseline_times
        mse = ((estimates - means) ** 2).mean(axis=(0, 2))
        adjusted_mse = mse * time_ratios

        _ = variances  # retained for future metric reporting extensions.
        performance_by_budget.append(float(np.mean(adjusted_mse)))

    return float(np.mean(performance_by_budget))
