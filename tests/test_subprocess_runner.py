import pytest
import whest as we

from whestbench.generation import sample_mlp
from whestbench.runner import (
    EstimatorEntrypoint,
    ResourceLimits,
    SubprocessRunner,
)
from whestbench.sdk import SetupContext


@pytest.fixture
def small_mlp():
    return sample_mlp(width=8, depth=2, rng=we.random.default_rng(42))


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


def test_subprocess_runner_stores_budget_breakdown(small_mlp, tmp_path) -> None:
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import whest as we\n"
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        base = we.zeros((mlp.depth, mlp.width), dtype=we.float32)\n"
        "        with we.namespace('phase'):\n"
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
        "import whest as we\n"
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        return we.zeros((mlp.depth, mlp.width), dtype=we.float32) + 1.0\n"
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
        "import whest as we\n"
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        acc = we.zeros((mlp.depth, mlp.width), dtype=we.float32)\n"
        "        with we.namespace('phase'):\n"
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
    with pytest.raises(we.BudgetExhaustedError):
        runner.predict(small_mlp, budget=50)
    stats = runner.last_predict_stats()
    assert stats is not None
    assert stats.budget_breakdown is not None
    assert "phase" in stats.budget_breakdown["by_namespace"]
    assert stats.budget_breakdown["by_namespace"]["phase"]["flops_used"] > 0
    runner.close()
