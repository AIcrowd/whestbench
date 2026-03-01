from collections.abc import Iterable as IterableABC
from dataclasses import FrozenInstanceError
from typing import get_args, get_origin, get_type_hints

import numpy as np
import pytest

from circuit_estimation import BaseEstimator, Circuit, SetupContext


class _RecordingEstimator(BaseEstimator):
    def __init__(self) -> None:
        self.calls: list[Circuit] = []

    def predict(self, circuit: Circuit, budget: int) -> np.ndarray:
        self.calls.append(circuit)
        return np.array([float(circuit.n), float(budget)], dtype=np.float32)


def test_base_estimator_default_predict_batch_stacks_predict_outputs() -> None:
    estimator = _RecordingEstimator()
    circuits = [
        Circuit(n=1, d=0, gates=[]),
        Circuit(n=2, d=0, gates=[]),
        Circuit(n=3, d=0, gates=[]),
    ]

    result = estimator.predict_batch(circuits, budget=11)

    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1.0, 11.0],
                [2.0, 11.0],
                [3.0, 11.0],
            ],
            dtype=np.float32,
        ),
    )
    assert estimator.calls == circuits


def test_base_estimator_predict_signatures_are_typed_to_circuit() -> None:
    predict_hints = get_type_hints(BaseEstimator.predict)
    assert predict_hints["circuit"] is Circuit

    batch_hints = get_type_hints(BaseEstimator.predict_batch)
    circuits_hint = batch_hints["circuits"]
    assert get_args(circuits_hint) == (Circuit,)
    assert get_origin(circuits_hint) is IterableABC


def test_setup_context_is_immutable_and_contains_required_fields() -> None:
    context = SetupContext(
        width=8,
        max_depth=12,
        budgets=(16, 32, 64),
        time_tolerance=0.05,
        api_version="1.0",
    )

    assert context.width == 8
    assert context.max_depth == 12
    assert context.budgets == (16, 32, 64)
    assert context.time_tolerance == 0.05
    assert context.api_version == "1.0"
    assert context.scratch_dir is None

    with pytest.raises(FrozenInstanceError):
        setattr(context, "width", 9)
