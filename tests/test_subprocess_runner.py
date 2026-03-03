from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np

from circuit_estimation.domain import Circuit
from circuit_estimation.runner import (
    DepthRowOutcome,
    EstimatorEntrypoint,
    ResourceLimits,
    SubprocessRunner,
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


def _context() -> SetupContext:
    return SetupContext(
        width=2,
        max_depth=1,
        budgets=(10,),
        time_tolerance=0.1,
        api_version="1.0",
    )


def test_subprocess_runner_streams_depth_rows(tmp_path: Path) -> None:
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
    runner = SubprocessRunner()
    runner.start(
        EstimatorEntrypoint(file_path=module_path), _context(),
        ResourceLimits(setup_timeout_s=2.0, predict_timeout_s=5.0, memory_limit_mb=256),
    )
    outcomes = list(runner.predict(_sample_circuit(), budget=10))
    runner.close()

    assert len(outcomes) == 1  # d=1
    assert outcomes[0].status == "ok"
    assert outcomes[0].depth_index == 0
    assert outcomes[0].row is not None
    np.testing.assert_allclose(outcomes[0].row, [0.0, 0.0])
    assert outcomes[0].wall_time_s >= 0.0


def test_subprocess_runner_times_out_predict_calls_and_returns_error(
    tmp_path: Path,
) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import time
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int):
                time.sleep(0.2)
                for _ in range(circuit.d):
                    yield np.zeros((circuit.n,), dtype=np.float32)
        """,
    )
    runner = SubprocessRunner()
    runner.start(
        EstimatorEntrypoint(file_path=module_path),
        _context(),
        ResourceLimits(
            setup_timeout_s=1.0,
            predict_timeout_s=0.05,
            memory_limit_mb=256,
        ),
    )

    outcomes = list(runner.predict(_sample_circuit(), budget=10))
    runner.close()

    assert len(outcomes) >= 1
    # At least one outcome should be an error with timeout message
    error_outcomes = [o for o in outcomes if o.status == "error"]
    assert len(error_outcomes) >= 1
    assert any("timed out" in (o.error_message or "") for o in error_outcomes)


def test_subprocess_runner_reports_protocol_errors_cleanly(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int):
                return 3.14
        """,
    )
    runner = SubprocessRunner()
    runner.start(
        EstimatorEntrypoint(file_path=module_path),
        _context(),
        ResourceLimits(
            setup_timeout_s=1.0,
            predict_timeout_s=1.0,
            memory_limit_mb=256,
        ),
    )
    outcomes = list(runner.predict(_sample_circuit(), budget=10))
    runner.close()

    assert len(outcomes) >= 1
    error_outcomes = [o for o in outcomes if o.status == "error"]
    assert len(error_outcomes) >= 1
    assert any("iterator" in (o.error_message or "") for o in error_outcomes)
