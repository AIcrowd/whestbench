import threading

import flopscope as flops
import flopscope.numpy as fnp
import pytest

import whestbench.runner as runner_module
from whestbench.generation import sample_mlp
from whestbench.runner import (
    EstimatorEntrypoint,
    ResourceLimits,
    SubprocessRunner,
)
from whestbench.sdk import SetupContext


def _limits(*, predict_timeout_s: float, wall_time_limit_s: float | None) -> ResourceLimits:
    return ResourceLimits(
        setup_timeout_s=10.0,
        predict_timeout_s=predict_timeout_s,
        memory_limit_mb=4096,
        flop_budget=100_000_000,
        wall_time_limit_s=wall_time_limit_s,
    )


@pytest.fixture
def small_mlp():
    return sample_mlp(width=8, depth=2, rng=fnp.random.default_rng(42))


def test_predict_response_timeout_adds_grace_to_wall_limit() -> None:
    timeout = getattr(runner_module, "_predict_response_timeout_s")

    assert timeout(_limits(predict_timeout_s=0.1, wall_time_limit_s=2.0)) == 7.0


def test_predict_response_timeout_keeps_longer_predict_timeout() -> None:
    timeout = getattr(runner_module, "_predict_response_timeout_s")

    assert timeout(_limits(predict_timeout_s=10.0, wall_time_limit_s=2.0)) == 10.0


def test_predict_response_timeout_without_wall_limit_uses_predict_timeout() -> None:
    timeout = getattr(runner_module, "_predict_response_timeout_s")

    assert timeout(_limits(predict_timeout_s=0.1, wall_time_limit_s=None)) == 0.1


def test_predict_response_timeout_nonfinite_wall_limit_uses_predict_timeout() -> None:
    timeout = getattr(runner_module, "_predict_response_timeout_s")

    assert timeout(_limits(predict_timeout_s=0.1, wall_time_limit_s=float("inf"))) == 0.1


def test_predict_response_timeout_oversized_wall_limit_uses_predict_timeout() -> None:
    timeout = getattr(runner_module, "_predict_response_timeout_s")

    assert (
        timeout(
            _limits(
                predict_timeout_s=0.1,
                wall_time_limit_s=threading.TIMEOUT_MAX * 2.0,
            )
        )
        == 0.1
    )


def test_subprocess_runner_predict_uses_derived_response_timeout(small_mlp, monkeypatch) -> None:
    runner = SubprocessRunner()
    recorded_timeouts: list[float] = []

    def read_response(timeout_s: float) -> dict[str, object]:
        recorded_timeouts.append(timeout_s)
        return {
            "status": "ok",
            "predictions": [[0.0] * small_mlp.width for _ in range(small_mlp.depth)],
        }

    monkeypatch.setattr(runner, "_started", True)
    monkeypatch.setattr(runner, "_process", object())
    monkeypatch.setattr(runner, "_limits", _limits(predict_timeout_s=0.1, wall_time_limit_s=2.0))
    monkeypatch.setattr(runner, "_send_request", lambda _payload: None)
    monkeypatch.setattr(runner, "_read_response", read_response)

    result = runner.predict(small_mlp, budget=100_000_000)

    assert recorded_timeouts == [7.0]
    assert result.shape == (2, 8)


def test_subprocess_runner_returns_timing(small_mlp, tmp_path) -> None:
    """Subprocess predict response includes timing fields in the parent BudgetContext."""
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import numpy as np\n"
        "from whestbench.sdk import BaseEstimator\n"
        "from whestbench.domain import MLP\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        return np.zeros((mlp.depth, mlp.width), dtype=np.float32)\n"
    )
    runner = SubprocessRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    limits = ResourceLimits(
        setup_timeout_s=10.0,
        predict_timeout_s=10.0,
        memory_limit_mb=4096,
        flop_budget=100_000_000,
    )
    runner.start(entry, ctx, limits)
    result = runner.predict(small_mlp, budget=100_000_000)
    assert result.shape == (2, 8)
    runner.close()


def test_subprocess_runner_predict(small_mlp, tmp_path) -> None:
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import numpy as np\n"
        "from whestbench.sdk import BaseEstimator\n"
        "from whestbench.domain import MLP\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        return np.zeros((mlp.depth, mlp.width), dtype=np.float32)\n"
    )
    runner = SubprocessRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    limits = ResourceLimits(
        setup_timeout_s=10.0, predict_timeout_s=10.0, memory_limit_mb=4096, flop_budget=100_000_000
    )
    runner.start(entry, ctx, limits)
    result = runner.predict(small_mlp, budget=100)
    assert result.shape == (2, 8)
    runner.close()


def test_subprocess_runner_waits_for_wall_limit_before_transport_timeout(
    small_mlp, tmp_path
) -> None:
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import time\n"
        "import numpy as np\n"
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        time.sleep(0.1)\n"
        "        return np.zeros((mlp.depth, mlp.width), dtype=np.float32)\n"
    )
    runner = SubprocessRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    limits = _limits(predict_timeout_s=0.05, wall_time_limit_s=0.5)
    runner.start(entry, ctx, limits)

    result = runner.predict(small_mlp, budget=100_000_000)

    assert result.shape == (2, 8)
    runner.close()


def test_subprocess_runner_stores_budget_breakdown(small_mlp, tmp_path) -> None:
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import flopscope as flops\nimport flopscope.numpy as fnp\n"
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        base = fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)\n"
        "        with flops.namespace('phase'):\n"
        "            return base + 1.0\n"
    )
    runner = SubprocessRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    limits = ResourceLimits(
        setup_timeout_s=10.0,
        predict_timeout_s=10.0,
        memory_limit_mb=4096,
        flop_budget=100_000_000,
    )
    runner.start(entry, ctx, limits)
    result = runner.predict(small_mlp, budget=100_000_000)
    assert result.shape == (2, 8)
    stats = runner.last_predict_stats()
    assert stats is not None
    assert stats.budget_breakdown is not None
    assert "phase" in stats.budget_breakdown["by_namespace"]
    assert stats.budget_breakdown["by_namespace"]["phase"]["flops_used"] > 0
    runner.close()


def test_subprocess_runner_stores_budget_breakdown_for_unlabeled_ops(small_mlp, tmp_path) -> None:
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import flopscope as flops\nimport flopscope.numpy as fnp\n"
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        return fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32) + 1.0\n"
    )
    runner = SubprocessRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    limits = ResourceLimits(
        setup_timeout_s=10.0,
        predict_timeout_s=10.0,
        memory_limit_mb=4096,
        flop_budget=100_000_000,
    )
    runner.start(entry, ctx, limits)
    runner.predict(small_mlp, budget=100_000_000)
    stats = runner.last_predict_stats()
    assert stats is not None
    assert stats.budget_breakdown is not None
    assert "null" in stats.budget_breakdown["by_namespace"]
    assert stats.budget_breakdown["by_namespace"]["null"]["calls"] >= 1
    runner.close()


def test_subprocess_runner_preserves_partial_budget_breakdown_on_exhaustion(
    small_mlp, tmp_path
) -> None:
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import flopscope as flops\nimport flopscope.numpy as fnp\n"
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        acc = fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)\n"
        "        with flops.namespace('phase'):\n"
        "            for _ in range(20):\n"
        "                acc = acc + 1.0\n"
        "        return acc\n"
    )
    runner = SubprocessRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    limits = ResourceLimits(
        setup_timeout_s=10.0,
        predict_timeout_s=10.0,
        memory_limit_mb=4096,
        flop_budget=100_000_000,
    )
    runner.start(entry, ctx, limits)
    with pytest.raises(flops.BudgetExhaustedError):
        runner.predict(small_mlp, budget=50)
    stats = runner.last_predict_stats()
    assert stats is not None
    assert stats.budget_breakdown is not None
    assert "phase" in stats.budget_breakdown["by_namespace"]
    assert stats.budget_breakdown["by_namespace"]["phase"]["flops_used"] > 0
    runner.close()
