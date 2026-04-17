import pytest
import whest as we

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
    return sample_mlp(width=8, depth=2, rng=we.random.default_rng(42))


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
    assert result.dtype == we.float32
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
