"""Tests for the line-protocol subprocess worker.

Most assertions drive the worker's functions in-process to stay fast;
a single end-to-end spawn validates the full stdio pipeline.
"""

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from whestbench import subprocess_worker


def _capture_responses(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Redirect `_write_response` into an in-memory list."""
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(subprocess_worker, "_write_response", captured.append)
    return captured


def _broken_predict_estimator(tmp_path: Path, message: str = "worker boom") -> Path:
    path = tmp_path / "broken_predict.py"
    path.write_text(
        dedent(
            f"""
            from whestbench import BaseEstimator

            class Estimator(BaseEstimator):
                def predict(self, mlp, budget):
                    raise RuntimeError({message!r})
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return path


def _broken_setup_estimator(tmp_path: Path, message: str = "setup boom") -> Path:
    path = tmp_path / "broken_setup.py"
    path.write_text(
        dedent(
            f"""
            from whestbench import BaseEstimator

            class Estimator(BaseEstimator):
                def setup(self, context):
                    raise RuntimeError({message!r})
                def predict(self, mlp, budget):
                    return None
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return path


def _wrong_shape_predict_estimator(tmp_path: Path) -> Path:
    path = tmp_path / "wrong_shape_predict.py"
    path.write_text(
        dedent(
            """
            from whestbench import BaseEstimator
            import flopscope as flops
            import flopscope.numpy as fnp

            class Estimator(BaseEstimator):
                def predict(self, mlp, budget):
                    return fnp.zeros((mlp.width, mlp.depth), dtype=fnp.float32)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return path


def _trivial_predict_request(width: int = 4, depth: int = 2) -> dict[str, Any]:
    weights = [
        [[1.0 if i == j else 0.0 for j in range(width)] for i in range(width)] for _ in range(depth)
    ]
    return {
        "command": "predict",
        "budget": 1_000_000,
        "mlp": {"width": width, "depth": depth, "weights": weights},
    }


def test_handle_predict_error_includes_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Driving `_handle_predict` in-process should emit a traceback that
    references the estimator source file."""
    from whestbench.loader import load_estimator_from_path

    estimator_path = _broken_predict_estimator(tmp_path)
    estimator, _ = load_estimator_from_path(estimator_path, class_name=None)

    captured = _capture_responses(monkeypatch)
    subprocess_worker._handle_predict(estimator, _trivial_predict_request())

    assert len(captured) == 1
    resp = captured[0]
    assert resp["status"] == "error"
    assert "worker boom" in resp["error_message"]
    assert isinstance(resp.get("traceback"), str)
    assert "RuntimeError" in resp["traceback"]
    assert "broken_predict.py" in resp["traceback"]


def test_handle_predict_shape_error_includes_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from whestbench.loader import load_estimator_from_path

    estimator_path = _wrong_shape_predict_estimator(tmp_path)
    estimator, _ = load_estimator_from_path(estimator_path, class_name=None)

    captured = _capture_responses(monkeypatch)
    subprocess_worker._handle_predict(estimator, _trivial_predict_request())

    assert len(captured) == 1
    resp = captured[0]
    assert resp["status"] == "error"
    assert "shape" in str(resp["error_message"]).lower()
    assert isinstance(resp["details"], dict)
    assert resp["details"]["expected_shape"] == [2, 4]
    assert resp["details"]["got_shape"] == [4, 2]


def test_handle_predict_budgets_exhaustion_errors_win_over_validation_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If budget/time exceptions inherit from ValueError, dedicated branches must win."""

    class FauxBudgetExhaustedError(ValueError):
        pass

    class FauxTimeExhaustedError(ValueError):
        pass

    monkeypatch.setattr(subprocess_worker.flops, "BudgetExhaustedError", FauxBudgetExhaustedError)
    monkeypatch.setattr(subprocess_worker.flops, "TimeExhaustedError", FauxTimeExhaustedError)

    class _BudgetEstimator:
        def predict(self, mlp, budget):  # type: ignore[no-untyped-def]
            raise FauxBudgetExhaustedError("faux budget")

    class _TimeEstimator:
        def predict(self, mlp, budget):  # type: ignore[no-untyped-def]
            raise FauxTimeExhaustedError("faux time")

    captured = _capture_responses(monkeypatch)
    subprocess_worker._handle_predict(_BudgetEstimator(), _trivial_predict_request())
    assert captured[-1]["status"] == "budget_exhausted"
    assert captured[-1]["error_message"] == "FLOP budget exceeded."

    captured = _capture_responses(monkeypatch)
    subprocess_worker._handle_predict(_TimeEstimator(), _trivial_predict_request())
    assert captured[-1]["status"] == "time_exhausted"
    assert captured[-1]["error_message"] == "Wall-clock time limit exceeded."


def test_main_start_error_includes_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`start` command with a setup-time exception should surface a traceback."""
    estimator_path = _broken_setup_estimator(tmp_path)
    start_request = {
        "command": "start",
        "entrypoint": {"file_path": str(estimator_path), "class_name": None},
        "context": {
            "width": 4,
            "depth": 2,
            "flop_budget": 1_000_000,
            "api_version": "1.0",
            "scratch_dir": None,
        },
        "wall_time_limit_s": None,
    }
    # Feed: start + close (close ends the loop).
    fake_stdin = io.StringIO(
        json.dumps(start_request) + "\n" + json.dumps({"command": "close"}) + "\n"
    )
    monkeypatch.setattr(subprocess_worker.sys, "stdin", fake_stdin)
    captured = _capture_responses(monkeypatch)

    rc = subprocess_worker.main()

    assert rc == 0
    # start_response first, close_response second.
    start_resp = captured[0]
    assert start_resp["status"] == "runtime_error"
    assert "setup boom" in start_resp["error_message"]
    assert isinstance(start_resp.get("traceback"), str)
    assert "RuntimeError" in start_resp["traceback"]
    assert "broken_setup.py" in start_resp["traceback"]


def test_worker_end_to_end_spawn_still_works(tmp_path: Path) -> None:
    """One slow end-to-end test: spawn a real subprocess and verify the stdio
    round-trip still produces a traceback in the predict-error path. The other
    traceback assertions run in-process above to keep the suite fast."""
    estimator_path = _broken_predict_estimator(tmp_path)
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path
    proc = subprocess.Popen(
        [sys.executable, "-m", "whestbench.subprocess_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdin is not None
    start = {
        "command": "start",
        "entrypoint": {"file_path": str(estimator_path), "class_name": None},
        "context": {
            "width": 4,
            "depth": 2,
            "flop_budget": 1_000_000,
            "api_version": "1.0",
            "scratch_dir": None,
        },
        "wall_time_limit_s": None,
    }
    proc.stdin.write(json.dumps(start) + "\n")
    proc.stdin.write(json.dumps(_trivial_predict_request()) + "\n")
    proc.stdin.write(json.dumps({"command": "close"}) + "\n")
    proc.stdin.flush()
    out, _err = proc.communicate(timeout=10)
    responses = [json.loads(line) for line in out.splitlines() if line.strip()]

    assert responses[0]["status"] == "ok"  # start
    predict_resp = responses[1]
    assert predict_resp["status"] == "error"
    assert "worker boom" in predict_resp["error_message"]
    assert "broken_predict.py" in predict_resp["traceback"]
