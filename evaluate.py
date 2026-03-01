"""Legacy scoring API preserved for backward compatibility in starter-kit code."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from circuit import Circuit, empirical_mean, random_circuit, run_batched
from circuit_estimation.scoring import ContestParams, default_contest_params

T = TypeVar("T")


def profile_fn(fn: Callable[[], Iterator[T]]) -> Iterator[tuple[float, T]]:
    """Yield cumulative elapsed time for each iterator output."""
    import time

    start_time = time.time()
    for output in fn():
        yield time.time() - start_time, output


def sampling_baseline_time(n_samples: int, width: int, depth: int) -> list[float]:
    """Return per-depth baseline wall times for random sampling forward passes."""
    circuit = random_circuit(width, depth)
    inputs = np.random.choice([-1.0, 1.0], size=(n_samples, width)).astype(np.float16)
    return [elapsed for elapsed, _ in profile_fn(lambda: run_batched(circuit, inputs))]


def score_estimator(
    estimator: Callable[[Circuit, int], Iterator[NDArray[np.float32]]],
    n_circuits: int,
    n_samples: int,
    contest_params: ContestParams = default_contest_params,
) -> float:
    """Legacy scoring function preserving behavior expected by existing tests."""
    n = contest_params.width
    d = contest_params.max_depth
    tolerance = contest_params.time_tolerance

    circuits = [random_circuit(n, d) for _ in range(n_circuits)]
    means: NDArray[np.float32] = np.array(
        [list(empirical_mean(circuit, n_samples)) for circuit in circuits], dtype=np.float32
    )
    performance_by_budget: list[float] = []

    for budget in contest_params.budgets:
        baseline_times: NDArray[np.float32] = np.array(
            sampling_baseline_time(budget, n, d), dtype=np.float32
        )
        baseline_times = np.maximum(baseline_times, np.float32(1e-9))

        runtimes = np.zeros(d, dtype=np.float32)
        all_outputs: list[list[NDArray[np.float32]]] = []
        for circuit in circuits:
            outputs: list[NDArray[np.float32]] = []
            for i, (elapsed, output) in enumerate(profile_fn(lambda: estimator(circuit, budget))):
                baseline_time = float(baseline_times[i])
                effective_time = max(float(elapsed), (1.0 - tolerance) * baseline_time)
                effective_output = (
                    output
                    if elapsed <= baseline_time * (1.0 + tolerance)
                    else np.zeros_like(output, dtype=np.float32)
                )
                runtimes[i] += np.float32(effective_time)
                outputs.append(np.asarray(effective_output, dtype=np.float32))
            all_outputs.append(outputs)

        estimates = np.array(all_outputs, dtype=np.float32)
        average_times = runtimes / np.float32(n_circuits)
        time_ratios = average_times / baseline_times
        mse = ((estimates - means) ** 2).mean(axis=(0, 2))
        adjusted_mse = mse * time_ratios
        performance_by_budget.append(float(np.mean(adjusted_mse)))

    return float(np.mean(performance_by_budget))


__all__ = [
    "ContestParams",
    "default_contest_params",
    "profile_fn",
    "random_circuit",
    "sampling_baseline_time",
    "score_estimator",
]
