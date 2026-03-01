from dataclasses import FrozenInstanceError
from typing import SupportsFloat, cast

import numpy as np
import pytest

from circuit_estimation import BaseEstimator, SetupContext


class _RecordingEstimator(BaseEstimator):
    def __init__(self) -> None:
        self.calls: list[object] = []

    def predict(self, circuit: object, budget: int) -> np.ndarray:
        self.calls.append(circuit)
        return np.array([float(cast(SupportsFloat, circuit)), float(budget)], dtype=np.float32)


def test_base_estimator_default_predict_batch_stacks_predict_outputs() -> None:
    estimator = _RecordingEstimator()

    result = estimator.predict_batch([1, 2, 3], budget=11)

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
    assert estimator.calls == [1, 2, 3]


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
