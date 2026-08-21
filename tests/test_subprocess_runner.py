import threading
from pathlib import Path

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
    # predict() also needs these to tell "started" from "died and needs a restart";
    # a runner faking the started state has to fake them too.
    monkeypatch.setattr(
        runner, "_context", SetupContext(width=8, depth=2, flop_budget=1, api_version="1.0")
    )
    monkeypatch.setattr(runner, "_entrypoint", EstimatorEntrypoint(file_path=Path("est.py")))
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


def test_worker_death_does_not_cascade_to_later_mlps(small_mlp, tmp_path) -> None:
    """A worker that hard-dies must fail ONLY the MLP that killed it.

    Before the restart path existed, SubprocessRunner.predict killed the process and
    left _started clear with no way back, so every later MLP hit the corpse with
    WORKER_BROKEN_PIPE. One OOM on MLP #1 of a 100-MLP suite was therefore scored as
    100 zero-prediction failures instead of 1 — and each cascaded MLP took the forced
    1.0 multiplier rather than the compute discount, despite never running. The
    contest rules scope the zero-prediction fallback to the MLP that actually failed.

    The estimator here dies exactly once, tracked through a file rather than instance
    state: a restarted worker is a fresh process with a fresh Estimator, so an
    in-memory counter would reset and every MLP would die again, hiding the fix.
    """
    marker = tmp_path / "died_once"
    est = tmp_path / "est_die_once.py"
    est.write_text(
        "import os\n"
        "import flopscope.numpy as fnp\n"
        "from whestbench.sdk import BaseEstimator\n"
        f"MARKER = {str(marker)!r}\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        if not os.path.exists(MARKER):\n"
        "            open(MARKER, 'w').close()\n"
        "            os._exit(1)\n"
        "        return fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)\n"
    )
    runner = SubprocessRunner()
    ctx = SetupContext(width=8, depth=2, flop_budget=100_000_000, api_version="1.0")
    runner.start(
        EstimatorEntrypoint(file_path=est),
        ctx,
        _limits(predict_timeout_s=30.0, wall_time_limit_s=30.0),
    )
    try:
        with pytest.raises(runner_module.RunnerError) as first:
            runner.predict(small_mlp, budget=100_000_000)
        assert first.value.detail.code in {"WORKER_EOF", "WORKER_BROKEN_PIPE"}

        # the whole point: the suite continues on a replacement worker
        for _ in range(2):
            out = runner.predict(small_mlp, budget=100_000_000)
            assert tuple(out.shape) == (small_mlp.depth, small_mlp.width)
    finally:
        runner.close()


def test_runner_not_started_still_raises_before_any_start() -> None:
    """The restart path must not turn a never-started runner into a silent restart."""
    runner = SubprocessRunner()
    mlp = sample_mlp(width=4, depth=2, rng=fnp.random.default_rng(0))
    with pytest.raises(runner_module.RunnerError) as exc:
        runner.predict(mlp, budget=1000)
    assert exc.value.detail.code == "RUNNER_NOT_STARTED"
