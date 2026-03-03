from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np

from circuit_estimation.domain import Circuit
from circuit_estimation.runner import (
    DepthRowOutcome,
    EstimatorEntrypoint,
    InProcessRunner,
    ResourceLimits,
)
from circuit_estimation.sdk import SetupContext
from tests.helpers import make_circuit, make_layer


def _write_estimator_module(tmp_path: Path, body: str) -> Path:
    module_path = tmp_path / "submission_estimator.py"
    module_path.write_text(dedent(body).strip() + "\n", encoding="utf-8")
    return module_path


def _sample_circuit() -> Circuit:
    return make_circuit(
        2,
        [
            make_layer(
                first=[0, 1],
                second=[1, 0],
                first_coeff=[1.0, 1.0],
                second_coeff=[0.0, 0.0],
                const=[0.0, 0.0],
                product_coeff=[0.0, 0.0],
            )
        ],
    )


def _setup_context() -> SetupContext:
    return SetupContext(
        width=2,
        max_depth=1,
        budgets=(10, 100),
        time_tolerance=0.1,
        api_version="1.0",
    )


def _limits() -> ResourceLimits:
    return ResourceLimits(
        setup_timeout_s=2.0,
        predict_timeout_s=1.0,
        memory_limit_mb=256,
        cpu_time_limit_s=None,
    )


def test_inprocess_runner_streams_depth_rows(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int):
                for i in range(circuit.d):
                    yield np.full((circuit.n,), float(i), dtype=np.float32)
        """,
    )
    runner = InProcessRunner()
    runner.start(EstimatorEntrypoint(file_path=module_path), _setup_context(), _limits())
    outcomes = list(runner.predict(_sample_circuit(), budget=10))

    assert len(outcomes) == 1  # _sample_circuit has d=1
    assert outcomes[0].status == "ok"
    assert outcomes[0].depth_index == 0
    assert outcomes[0].row is not None
    np.testing.assert_allclose(outcomes[0].row, [0.0, 0.0])
    assert outcomes[0].wall_time_s >= 0.0


def test_inprocess_runner_calls_setup_once_before_predicts(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            setup_calls = 0

            def setup(self, context):
                type(self).setup_calls += 1

            def predict(self, circuit: Circuit, budget: int):
                for _ in range(circuit.d):
                    yield np.full((circuit.n,), float(type(self).setup_calls), dtype=np.float32)
        """,
    )

    runner = InProcessRunner()
    runner.start(EstimatorEntrypoint(file_path=module_path), _setup_context(), _limits())
    first = list(runner.predict(_sample_circuit(), budget=10))
    second = list(runner.predict(_sample_circuit(), budget=100))

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].status == "ok"
    assert second[0].status == "ok"
    assert first[0].row is not None
    assert second[0].row is not None
    np.testing.assert_allclose(first[0].row, np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(second[0].row, np.ones(2, dtype=np.float32))


def test_inprocess_runner_streams_error_on_runtime_exception(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int):
                raise RuntimeError("boom")
        """,
    )

    runner = InProcessRunner()
    runner.start(EstimatorEntrypoint(file_path=module_path), _setup_context(), _limits())
    outcomes = list(runner.predict(_sample_circuit(), budget=10))

    assert len(outcomes) == 1
    assert outcomes[0].status == "error"
    assert outcomes[0].row is None
    assert outcomes[0].error_message is not None
    assert "boom" in outcomes[0].error_message


def test_inprocess_runner_streams_cumulative_wall_times(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import time
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int):
                for _ in range(circuit.d):
                    time.sleep(0.01)
                    yield np.zeros((circuit.n,), dtype=np.float32)
        """,
    )
    ctx = SetupContext(width=2, max_depth=3, budgets=(10,), time_tolerance=0.1, api_version="1.0")
    circuit = make_circuit(
        2,
        [
            make_layer(
                first=[0, 1], second=[1, 0],
                first_coeff=[1.0, 1.0], second_coeff=[0.0, 0.0],
                const=[0.0, 0.0], product_coeff=[0.0, 0.0],
            )
        ] * 3,
    )
    runner = InProcessRunner()
    runner.start(EstimatorEntrypoint(file_path=module_path), ctx, _limits())
    outcomes = list(runner.predict(circuit, budget=10))

    assert len(outcomes) == 3
    # wall times should be monotonically increasing (cumulative)
    for i in range(1, len(outcomes)):
        assert outcomes[i].wall_time_s >= outcomes[i - 1].wall_time_s
