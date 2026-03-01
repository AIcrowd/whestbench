from __future__ import annotations

import numpy as np
import pytest

from circuit_estimation.domain import Circuit, Layer
from circuit_estimation.scoring import ContestParams, score_estimator


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


def test_score_estimator_rejects_wrong_output_width() -> None:
    # Contract check: participant estimators must emit one value per wire.
    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(n=2, d=1, value=1.0)

    def bad_estimator(_circuit: Circuit, _budget: int):
        yield np.array([0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="output width"):
        score_estimator(
            bad_estimator,
            n_circuits=1,
            n_samples=4,
            contest_params=params,
            circuits=[circuit],
        )


def test_score_estimator_profile_hook_collects_runtime_cpu_and_memory() -> None:
    # Observability check: profiler payload includes core runtime/resource diagnostics.
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
    assert {"wall_time_s", "cpu_time_s", "rss_bytes", "peak_rss_bytes"} <= set(last.keys())


def test_score_estimator_raises_when_estimator_stops_early() -> None:
    # Regression check: yielding fewer layers than max_depth should fail loudly.
    params = ContestParams(width=2, max_depth=2, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(n=2, d=2, value=1.0)

    def short_estimator(_circuit: Circuit, _budget: int):
        yield np.array([1.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="expected 2"):
        score_estimator(
            short_estimator,
            n_circuits=1,
            n_samples=4,
            contest_params=params,
            circuits=[circuit],
        )
