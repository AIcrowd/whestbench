import flopscope.numpy as fnp
import pytest

from whestbench.generation import sample_mlp
from whestbench.runner import (
    EstimatorEntrypoint,
    LocalRunner,
    ResourceLimits,
    RunnerError,
)
from whestbench.sdk import SetupContext


@pytest.fixture
def small_mlp():
    return sample_mlp(width=8, depth=2, rng=fnp.random.default_rng(42))


@pytest.fixture
def limits():
    return ResourceLimits(
        setup_timeout_s=5.0, predict_timeout_s=30.0, memory_limit_mb=4096, flop_budget=1_000_000
    )


def test_inprocess_runner_predict_returns_array(small_mlp, limits, tmp_path) -> None:
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import numpy as np\n"
        "from whestbench.sdk import BaseEstimator\n"
        "from whestbench.domain import MLP\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        return np.zeros((mlp.depth, mlp.width), dtype=np.float32)\n"
    )
    runner = LocalRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    runner.start(entry, ctx, limits)
    result = runner.predict(small_mlp, budget=100)
    assert result.shape == (2, 8)
    assert result.dtype == fnp.float32
    runner.close()


def test_inprocess_runner_predict_skips_validation(small_mlp, limits, tmp_path) -> None:
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import numpy as np\n"
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        arr = np.zeros((mlp.depth, mlp.width), dtype=np.float32)\n"
        "        arr[0, 0] = np.inf\n"
        "        return arr\n"
    )
    runner = LocalRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    runner.start(entry, ctx, limits)
    result = runner.predict(small_mlp, budget=100)
    assert result.shape == (2, 8)
    assert float(result[0, 0]) == float("inf")
    runner.close()


def test_inprocess_runner_predict_before_start_raises(small_mlp) -> None:
    runner = LocalRunner()
    with pytest.raises(RunnerError):
        runner.predict(small_mlp, budget=100)


def test_inprocess_runner_predict_preserves_estimator_error_details(
    small_mlp, limits, tmp_path
) -> None:
    details = {
        "expected_shape": [2, 8],
        "got_shape": [8, 2],
        "hint": "Returned predictions appear to be transposed: expected (depth, width), got (width, depth).",
        "cause_hints": [
            "Returned predictions appear to be transposed: expected (depth, width), got (width, depth)."
        ],
    }
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import numpy as np\n"
        "from whestbench.sdk import BaseEstimator\n"
        "from whestbench.domain import MLP\n"
        "\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        exc = ValueError(f'Predictions must have shape ({mlp.depth}, {mlp.width}), got ({mlp.width}, {mlp.depth}).')\n"
        "        exc.details = {'expected_shape': [2, 8], 'got_shape': [8, 2], 'hint': 'Returned predictions appear to be transposed: expected (depth, width), got (width, depth).', 'cause_hints': ['Returned predictions appear to be transposed: expected (depth, width), got (width, depth).']}\n"
        "        raise exc\n"
    )
    runner = LocalRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    runner.start(entry, ctx, limits)
    with pytest.raises(RunnerError) as exc_info:
        runner.predict(small_mlp, budget=100)
    assert exc_info.value.detail.code == "PREDICT_ERROR"
    assert exc_info.value.detail.details == details
    runner.close()


def test_local_runner_preserves_time_exhausted_error(small_mlp, limits, tmp_path) -> None:
    """TimeExhaustedError must escape LocalRunner with its TYPE intact.

    scoring.py has a dedicated `except flops.TimeExhaustedError` handler that sets
    time_exhausted=True and buckets the MLP under failure_breakdown["time_exhausted"].
    LocalRunner.predict wraps unknown exceptions in RunnerError(PREDICT_ERROR); if
    TimeExhaustedError falls into that generic clause, the dedicated handler becomes
    unreachable and the very MLP that blew the wall cap reports time_exhausted=False
    through the generic error path. BudgetExhaustedError is re-raised for exactly the
    same reason, and SubprocessRunner preserves the type independently, so a
    regression here would also make the two runners disagree.
    """
    import flopscope as flops

    est_file = tmp_path / "est_time.py"
    est_file.write_text(
        "import flopscope as flops\n"
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        raise flops.TimeExhaustedError('predict', elapsed_s=9.0, limit_s=1.0)\n"
    )
    runner = LocalRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    runner.start(entry, ctx, limits)
    with pytest.raises(flops.TimeExhaustedError):
        runner.predict(small_mlp, budget=100)
    runner.close()


def test_local_runner_preserves_budget_exhausted_error(small_mlp, limits, tmp_path) -> None:
    """The sibling case, pinned alongside so the two cannot drift apart again."""
    import flopscope as flops

    est_file = tmp_path / "est_budget.py"
    est_file.write_text(
        "import flopscope as flops\n"
        "from whestbench.sdk import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        raise flops.BudgetExhaustedError('predict', flop_cost=9, flops_remaining=0)\n"
    )
    runner = LocalRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, flop_budget=100, api_version="1.0")
    runner.start(entry, ctx, limits)
    with pytest.raises(flops.BudgetExhaustedError):
        runner.predict(small_mlp, budget=100)
    runner.close()
