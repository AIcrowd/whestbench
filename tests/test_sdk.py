from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from circuit_estimation.domain import Circuit, Layer
from circuit_estimation.sdk import BaseEstimator


def _circuit(n: int, d: int) -> Circuit:
    layer = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1, 0], dtype=np.int32),
        first_coeff=np.zeros(n, dtype=np.float32),
        second_coeff=np.zeros(n, dtype=np.float32),
        const=np.zeros(n, dtype=np.float32),
        product_coeff=np.zeros(n, dtype=np.float32),
    )
    return Circuit(n=n, d=d, gates=[layer for _ in range(d)])


class ExampleEstimator(BaseEstimator):
    def predict(self, circuit: Circuit, budget: int) -> Iterator[np.ndarray]:
        for _ in range(circuit.d):
            yield np.zeros((circuit.n,), dtype=np.float32)


def test_predict_streaming_signature_and_depth_rows() -> None:
    est = ExampleEstimator()
    rows = list(est.predict(_circuit(2, 3), budget=10))
    assert len(rows) == 3
    assert all(row.shape == (2,) for row in rows)
