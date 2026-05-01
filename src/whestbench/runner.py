"""Runner interfaces for estimator isolation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback as _tb
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol

import flopscope as flops
import flopscope.numpy as fnp

from .domain import MLP
from .loader import load_estimator_from_path
from .sdk import BaseEstimator, SetupContext

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

try:
    import psutil  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover
    psutil = None

RunnerStage = Literal["load", "setup", "predict", "validate", "package", "submit"]


@dataclass(frozen=True)
class EstimatorEntrypoint:
    file_path: Path
    class_name: Optional[str] = None


@dataclass(frozen=True)
class ResourceLimits:
    setup_timeout_s: float
    predict_timeout_s: float
    memory_limit_mb: int
    flop_budget: int
    cpu_time_limit_s: Optional[float] = None
    wall_time_limit_s: Optional[float] = None
    untracked_time_limit_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.setup_timeout_s <= 0:
            raise ValueError("setup_timeout_s must be positive.")
        if self.predict_timeout_s <= 0:
            raise ValueError("predict_timeout_s must be positive.")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive.")
        if self.flop_budget <= 0:
            raise ValueError("flop_budget must be positive.")
        if self.cpu_time_limit_s is not None and self.cpu_time_limit_s <= 0:
            raise ValueError("cpu_time_limit_s must be positive when provided.")
        if self.wall_time_limit_s is not None and self.wall_time_limit_s <= 0:
            raise ValueError("wall_time_limit_s must be positive when provided.")
        if self.untracked_time_limit_s is not None and self.untracked_time_limit_s <= 0:
            raise ValueError("untracked_time_limit_s must be positive when provided.")


@dataclass(frozen=True)
class RunnerErrorDetail:
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None


class RunnerError(RuntimeError):
    def __init__(self, stage: str, detail: RunnerErrorDetail):
        super().__init__(detail.message)
        self.stage = stage
        self.detail = detail


@dataclass(frozen=True)
class PredictStats:
    # wall_time_s ≈ tracked_time_s + flopscope_overhead_time_s + untracked_time_s
    flops_used: int
    wall_time_s: float
    tracked_time_s: float
    flopscope_overhead_time_s: float
    untracked_time_s: float
    budget_breakdown: Optional[Dict[str, Any]] = None


class EstimatorRunner(Protocol):
    def start(
        self,
        entrypoint: EstimatorEntrypoint,
        context: SetupContext,
        limits: ResourceLimits,
    ) -> None: ...

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray: ...

    def close(self) -> None: ...


def _mlp_to_payload(mlp: MLP) -> Dict[str, Any]:
    return {
        "width": int(mlp.width),
        "depth": int(mlp.depth),
        "weights": [w.tolist() for w in mlp.weights],
    }


class LocalRunner:
    def __init__(self) -> None:
        self._estimator: Optional[BaseEstimator] = None
        self._limits: Optional[ResourceLimits] = None
        self._context: Optional[SetupContext] = None
        self._started = False
        self._last_predict_stats: Optional[PredictStats] = None

    def start(
        self,
        entrypoint: EstimatorEntrypoint,
        context: SetupContext,
        limits: ResourceLimits,
    ) -> None:
        self.close()
        self._limits = limits
        self._context = context
        start_wall = time.time()
        estimator, _ = load_estimator_from_path(
            entrypoint.file_path, class_name=entrypoint.class_name
        )
        self._estimator = estimator
        try:
            estimator.setup(context)
        except Exception as exc:
            setup_details = getattr(exc, "details", None)
            if not isinstance(setup_details, dict):
                setup_details = None
            raise RunnerError(
                "setup",
                RunnerErrorDetail(
                    code="SETUP_ERROR",
                    message=str(exc),
                    details=setup_details,
                    traceback=_tb.format_exc(),
                ),
            ) from exc
        setup_elapsed = time.time() - start_wall
        if setup_elapsed > limits.setup_timeout_s:
            raise RunnerError(
                "setup",
                RunnerErrorDetail(
                    code="SETUP_TIMEOUT",
                    message=f"setup exceeded timeout ({setup_elapsed:.6f}s > {limits.setup_timeout_s:.6f}s)",
                ),
            )
        self._started = True

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        if not self._started or self._estimator is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="RUNNER_NOT_STARTED",
                    message="Runner must be started before calling predict.",
                ),
            )
        try:
            return self._estimator.predict(mlp, budget)
        except flops.BudgetExhaustedError:
            raise
        except Exception as exc:
            predict_details = getattr(exc, "details", None)
            if not isinstance(predict_details, dict):
                predict_details = None
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="PREDICT_ERROR",
                    message=str(exc),
                    details=predict_details,
                    traceback=_tb.format_exc(),
                ),
            ) from exc

    def last_predict_stats(self) -> Optional[PredictStats]:
        return self._last_predict_stats

    def close(self) -> None:
        if self._estimator is not None:
            teardown = getattr(self._estimator, "teardown", None)
            if callable(teardown):
                teardown()
        self._estimator = None
        self._limits = None
        self._context = None
        self._started = False
        self._last_predict_stats = None


# Backward-compatible alias
InProcessRunner = LocalRunner


class SubprocessRunner:
    def __init__(self, *, worker_command: Optional[List[str]] = None) -> None:
        self._worker_command = (
            worker_command
            if worker_command is not None
            else [sys.executable, "-m", "whestbench.subprocess_worker"]
        )
        self._process: Optional[subprocess.Popen] = None
        self._limits: Optional[ResourceLimits] = None
        self._context: Optional[SetupContext] = None
        self._started = False
        self._last_predict_stats: Optional[PredictStats] = None
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._stderr_reader: Optional[threading.Thread] = None

    def start(
        self,
        entrypoint: EstimatorEntrypoint,
        context: SetupContext,
        limits: ResourceLimits,
    ) -> None:
        self.close()
        self._limits = limits
        self._context = context
        self._process = subprocess.Popen(
            self._worker_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._worker_env(),
        )
        self._stderr_reader = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
            name="SubprocessRunner-stderr",
        )
        self._stderr_reader.start()
        self._send_request(
            {
                "command": "start",
                "entrypoint": {
                    "file_path": str(entrypoint.file_path),
                    "class_name": entrypoint.class_name,
                },
                "context": {
                    "width": context.width,
                    "depth": context.depth,
                    "flop_budget": context.flop_budget,
                    "api_version": context.api_version,
                    "scratch_dir": context.scratch_dir,
                },
                "wall_time_limit_s": limits.wall_time_limit_s,
            }
        )
        try:
            response = self._read_response(timeout_s=limits.setup_timeout_s)
        except TimeoutError as exc:
            self._terminate_process()
            raise RunnerError(
                "setup",
                RunnerErrorDetail(code="SETUP_TIMEOUT", message="worker setup timed out."),
            ) from exc
        except RunnerError as exc:
            stderr_tail = self._read_stderr_tail()
            msg = exc.detail.message
            if stderr_tail:
                msg = f"{msg} stderr: {stderr_tail}"
            self._terminate_process()
            raise RunnerError(
                "setup",
                RunnerErrorDetail(code="SETUP_PROTOCOL_ERROR", message=msg),
            ) from exc
        if response.get("status") != "ok":
            raise RunnerError(
                "setup",
                RunnerErrorDetail(
                    code="SETUP_ERROR",
                    message=str(response.get("error_message", "worker setup failed")),
                    traceback=response.get("traceback"),
                ),
            )
        self._started = True

    def last_predict_stats(self) -> Optional[PredictStats]:
        return self._last_predict_stats

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        if not self._started or self._process is None or self._limits is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="RUNNER_NOT_STARTED", message="Runner must be started."),
            )
        self._last_predict_stats = None
        self._send_request(
            {
                "command": "predict",
                "budget": int(budget),
                "mlp": _mlp_to_payload(mlp),
            }
        )
        try:
            response = self._read_response(timeout_s=self._limits.predict_timeout_s)
        except TimeoutError:
            self._terminate_process()
            self._started = False
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="PREDICT_TIMEOUT", message="predict timed out."),
            )
        except RunnerError:
            raise

        self._last_predict_stats = PredictStats(
            flops_used=int(response.get("flops_used", 0)),
            wall_time_s=float(response.get("wall_time_s", 0.0) or 0.0),
            tracked_time_s=float(response.get("tracked_time_s", 0.0) or 0.0),
            flopscope_overhead_time_s=float(response.get("flopscope_overhead_time_s", 0.0) or 0.0),
            untracked_time_s=float(response.get("untracked_time_s", 0.0) or 0.0),
            budget_breakdown=response.get("budget_breakdown"),
        )
        if response.get("status") == "budget_exhausted":
            raise flops.BudgetExhaustedError("subprocess_predict", flop_cost=0, flops_remaining=0)
        if response.get("status") == "time_exhausted":
            raise flops.TimeExhaustedError("subprocess_predict", elapsed_s=0.0, limit_s=0.0)
        if response.get("status") == "error":
            details = response.get("details")
            if not isinstance(details, dict):
                details = None
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="PREDICT_ERROR",
                    message=str(response.get("error_message", "unknown error")),
                    traceback=response.get("traceback"),
                    details=details,
                ),
            )
        predictions_data = response.get("predictions")
        if predictions_data is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="PREDICT_NO_DATA", message="No predictions in response."),
            )
        result = fnp.asarray(predictions_data, dtype=fnp.float32)
        return result

    def close(self) -> None:
        if self._process is None:
            self._started = False
            return
        if self._process.poll() is None:
            try:
                self._send_request({"command": "close"})
                self._read_response(timeout_s=0.5)
            except Exception:
                pass
            self._terminate_process()
        if self._stderr_reader is not None and self._stderr_reader.is_alive():
            self._stderr_reader.join(timeout=0.1)
        self._process = None
        self._limits = None
        self._context = None
        self._started = False
        self._last_predict_stats = None

    def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            for line in self._process.stderr:
                self._stderr_lines.append(line.rstrip("\n"))
        except Exception:
            pass

    def _send_request(self, payload: Dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="WORKER_IO_ERROR", message="Worker stdin unavailable."),
            )
        try:
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="WORKER_BROKEN_PIPE", message="Worker stdin closed."),
            ) from exc

    def _read_response(self, timeout_s: float) -> Dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="WORKER_IO_ERROR", message="Worker stdout unavailable."),
            )

        result: List[str] = []

        def _read() -> None:
            assert self._process is not None and self._process.stdout is not None
            result.append(self._process.stdout.readline())

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(timeout=timeout_s)
        if reader.is_alive():
            raise TimeoutError("worker response timed out")
        if not result or result[0] == "":
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="WORKER_EOF", message="Worker closed stdout."),
            )
        try:
            payload = json.loads(result[0])
        except json.JSONDecodeError as exc:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="WORKER_PROTOCOL_ERROR", message="Invalid JSON."),
            ) from exc
        return payload

    def _terminate_process(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=1.0)

    def _read_stderr_tail(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        if self._stderr_lines:
            return self._stderr_lines[-1]
        if self._process.poll() is None:
            return ""
        stderr = self._process.stderr.read().strip()
        return stderr.splitlines()[-1] if stderr else ""

    def _worker_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parents[1])
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src_root if not current else f"{src_root}{os.pathsep}{current}"
        return env
