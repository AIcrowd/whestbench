from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np

from circuit_estimation.domain import Circuit
from circuit_estimation.runner import EstimatorEntrypoint, InProcessRunner, ResourceLimits
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


def test_inprocess_runner_calls_setup_once_before_predicts(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator

        class Estimator(BaseEstimator):
            setup_calls = 0

            def setup(self, context):
                type(self).setup_calls += 1

            def predict(self, circuit, budget: int):
                return np.full((circuit.d, circuit.n), float(type(self).setup_calls), dtype=np.float32)
        """,
    )

    runner = InProcessRunner()
    runner.start(EstimatorEntrypoint(file_path=module_path), _setup_context(), _limits())
    first = runner.predict(_sample_circuit(), budget=10)
    second = runner.predict(_sample_circuit(), budget=100)

    assert first.status == "ok"
    assert second.status == "ok"
    assert first.predictions is not None
    assert second.predictions is not None
    np.testing.assert_allclose(first.predictions, np.ones((1, 2), dtype=np.float32))
    np.testing.assert_allclose(second.predictions, np.ones((1, 2), dtype=np.float32))


def test_inprocess_runner_collects_wall_cpu_and_memory_metrics(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator

        class Estimator(BaseEstimator):
            def predict(self, circuit, budget: int):
                return np.zeros((circuit.d, circuit.n), dtype=np.float32)
        """,
    )
    runner = InProcessRunner()
    runner.start(EstimatorEntrypoint(file_path=module_path), _setup_context(), _limits())
    outcome = runner.predict(_sample_circuit(), budget=10)

    assert outcome.status == "ok"
    assert outcome.predictions is not None
    assert outcome.predictions.shape == (1, 2)
    assert outcome.wall_time_s >= 0.0
    assert outcome.cpu_time_s >= 0.0
    assert outcome.rss_bytes >= 0
    assert outcome.peak_rss_bytes >= 0


def test_inprocess_runner_returns_structured_runtime_error_status(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        from circuit_estimation import BaseEstimator

        class Estimator(BaseEstimator):
            def predict(self, circuit, budget: int):
                raise RuntimeError("boom")
        """,
    )

    runner = InProcessRunner()
    runner.start(EstimatorEntrypoint(file_path=module_path), _setup_context(), _limits())
    outcome = runner.predict(_sample_circuit(), budget=10)

    assert outcome.status == "runtime_error"
    assert outcome.predictions is None
    assert outcome.error_message is not None
    assert "boom" in outcome.error_message
