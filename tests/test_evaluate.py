import numpy as np
import pytest

import evaluate
from evaluate import ContestParams, profile_fn, sampling_baseline_time, score_estimator
from tests.helpers import make_circuit, make_layer


def _constant_circuit(n: int, d: int, value: float = 1.0):
    layers = []
    for _ in range(d):
        layers.append(
            make_layer(
                first=[0] * n,
                second=[1] * n,
                first_coeff=[0.0] * n,
                second_coeff=[0.0] * n,
                const=[value] * n,
                product_coeff=[0.0] * n,
            )
        )
    return make_circuit(n, layers)


def test_profile_fn_reports_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    time_values = iter([10.0, 10.1, 10.3, 10.6])

    class _FakeTime:
        @staticmethod
        def time() -> float:
            return next(time_values)

    monkeypatch.setitem(__import__("sys").modules, "time", _FakeTime)

    values = list(profile_fn(lambda: iter(["a", "b", "c"])))

    assert [v for _, v in values] == ["a", "b", "c"]
    np.testing.assert_allclose([t for t, _ in values], [0.1, 0.3, 0.6])


def test_sampling_baseline_time_returns_depth_entries() -> None:
    result = sampling_baseline_time(n_samples=8, width=4, depth=3)

    assert len(result) == 3
    assert all(time_value >= 0.0 for time_value in result)


def test_score_estimator_applies_timeout_zeroing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate, "random_circuit", lambda n, d: _constant_circuit(n=n, d=d, value=1.0))
    monkeypatch.setattr(evaluate, "sampling_baseline_time", lambda n_samples, width, depth: [1.0] * depth)

    def fake_profile_fn(fn):
        for output in fn():
            yield 1.2, output

    monkeypatch.setattr(evaluate, "profile_fn", fake_profile_fn)

    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)

    def estimator(_circuit, _budget):
        yield np.array([0.0, 0.0], dtype=np.float32)

    score = score_estimator(estimator, n_circuits=2, n_samples=4, contest_params=params)

    assert score == pytest.approx(1.2)


def test_score_estimator_applies_minimum_time_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate, "random_circuit", lambda n, d: _constant_circuit(n=n, d=d, value=1.0))
    monkeypatch.setattr(evaluate, "sampling_baseline_time", lambda n_samples, width, depth: [1.0] * depth)

    def fake_profile_fn(fn):
        for output in fn():
            yield 0.1, output

    monkeypatch.setattr(evaluate, "profile_fn", fake_profile_fn)

    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)

    def estimator(_circuit, _budget):
        yield np.array([0.0, 0.0], dtype=np.float32)

    score = score_estimator(estimator, n_circuits=3, n_samples=8, contest_params=params)

    assert score == pytest.approx(0.9)
