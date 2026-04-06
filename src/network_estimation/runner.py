"""Runner interfaces for estimator isolation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol

import mechestim as me

from .domain import MLP
from .loader import load_estimator_from_path
from .scoring import validate_predictions
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


class EstimatorRunner(Protocol):
    def start(
        self,
        entrypoint: EstimatorEntrypoint,
        context: SetupContext,
        limits: ResourceLimits,
    ) -> None: ...

    def predict(self, mlp: MLP, budget: int) -> me.ndarray: ...

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
            raise RunnerError(
                "setup",
                RunnerErrorDetail(code="SETUP_ERROR", message=str(exc)),
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

    def predict(self, mlp: MLP, budget: int) -> me.ndarray:
        if not self._started or self._estimator is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="RUNNER_NOT_STARTED",
                    message="Runner must be started before calling predict.",
                ),
            )
        try:
            with me.BudgetContext(flop_budget=budget):
                raw = self._estimator.predict(mlp, budget)
                return validate_predictions(raw, depth=mlp.depth, width=mlp.width)
        except me.BudgetExhaustedError as exc:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="BUDGET_EXHAUSTED", message=str(exc)),
            ) from exc

    def close(self) -> None:
        if self._estimator is not None:
            teardown = getattr(self._estimator, "teardown", None)
            if callable(teardown):
                teardown()
        self._estimator = None
        self._limits = None
        self._context = None
        self._started = False


# Backward-compatible alias
InProcessRunner = LocalRunner


class SubprocessRunner:
    def __init__(self, *, worker_command: Optional[List[str]] = None) -> None:
        self._worker_command = (
            worker_command
            if worker_command is not None
            else [sys.executable, "-m", "network_estimation.subprocess_worker"]
        )
        self._process: Optional[subprocess.Popen] = None
        self._limits: Optional[ResourceLimits] = None
        self._context: Optional[SetupContext] = None
        self._started = False

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
                ),
            )
        self._started = True

    def predict(self, mlp: MLP, budget: int) -> me.ndarray:
        if not self._started or self._process is None or self._limits is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="RUNNER_NOT_STARTED", message="Runner must be started."),
            )
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

        if response.get("status") == "error":
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="PREDICT_ERROR",
                    message=str(response.get("error_message", "unknown error")),
                ),
            )
        predictions_data = response.get("predictions")
        if predictions_data is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="PREDICT_NO_DATA", message="No predictions in response."),
            )
        return me.asarray(predictions_data, dtype=me.float32)

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
        self._process = None
        self._limits = None
        self._context = None
        self._started = False

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
        import threading

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
