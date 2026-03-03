"""Runner interfaces and structured execution outcomes for estimator isolation."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit
from .loader import load_estimator_from_path
from .sdk import BaseEstimator, SetupContext
from .streaming import validate_depth_row

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX environments
    resource = None

try:
    import psutil  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

PredictStatus = Literal["ok", "timeout", "oom", "runtime_error", "protocol_error"]
RunnerStage = Literal["load", "setup", "predict", "validate", "package", "submit"]


@dataclass(frozen=True, slots=True)
class EstimatorEntrypoint:
    """Location of participant estimator implementation."""

    file_path: Path
    class_name: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Runtime limits applied by concrete runner implementations."""

    setup_timeout_s: float
    predict_timeout_s: float
    memory_limit_mb: int
    cpu_time_limit_s: float | None = None

    def __post_init__(self) -> None:
        if self.setup_timeout_s <= 0:
            raise ValueError("setup_timeout_s must be positive.")
        if self.predict_timeout_s <= 0:
            raise ValueError("predict_timeout_s must be positive.")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive.")
        if self.cpu_time_limit_s is not None and self.cpu_time_limit_s <= 0:
            raise ValueError("cpu_time_limit_s must be positive when provided.")


@dataclass(slots=True)
class PredictOutcome:
    """Per-call prediction outcome returned by estimator runners."""

    predictions: NDArray[np.float32] | None
    wall_time_s: float
    cpu_time_s: float
    rss_bytes: int
    peak_rss_bytes: int
    status: PredictStatus
    error_message: str | None = None

    def __post_init__(self) -> None:
        valid_status = {"ok", "timeout", "oom", "runtime_error", "protocol_error"}
        if self.status not in valid_status:
            raise ValueError(f"Unsupported status: {self.status}")
        if self.wall_time_s < 0:
            raise ValueError("wall_time_s must be non-negative.")
        if self.cpu_time_s < 0:
            raise ValueError("cpu_time_s must be non-negative.")
        if self.rss_bytes < 0:
            raise ValueError("rss_bytes must be non-negative.")
        if self.peak_rss_bytes < 0:
            raise ValueError("peak_rss_bytes must be non-negative.")
        if self.status == "ok" and self.predictions is None:
            raise ValueError("predictions are required when status is 'ok'.")


DepthRowStatus = Literal["ok", "error"]


@dataclass(slots=True)
class DepthRowOutcome:
    """Per-depth-row outcome yielded by streaming runner predict."""

    depth_index: int
    row: NDArray[np.float32] | None
    wall_time_s: float
    status: DepthRowStatus
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class RunnerErrorDetail:
    """Machine-readable error payload surfaced to CLI and report layers."""

    code: str
    message: str
    details: dict[str, str | int | float | bool] | None = None
    traceback: str | None = None


class RunnerError(RuntimeError):
    """Structured runner exception carrying stable stage + error detail."""

    def __init__(self, stage: RunnerStage, detail: RunnerErrorDetail):
        super().__init__(detail.message)
        self.stage = stage
        self.detail = detail


class EstimatorRunner(Protocol):
    """Protocol for in-process, subprocess, and future cloud runners."""

    def start(
        self,
        entrypoint: EstimatorEntrypoint,
        context: SetupContext,
        limits: ResourceLimits,
    ) -> None: ...

    def predict(self, circuit: Circuit, budget: int) -> PredictOutcome: ...

    def predict_batch(self, circuits: list[Circuit], budget: int) -> list[PredictOutcome]: ...

    def close(self) -> None: ...


def _peak_rss_bytes() -> int:
    if resource is None:
        return 0
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return usage
    return usage * 1024


def _rss_bytes() -> int:
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)
    return _peak_rss_bytes()


def _circuit_to_payload(circuit: Circuit) -> dict[str, Any]:
    gates: list[dict[str, list[float] | list[int]]] = []
    for layer in circuit.gates:
        gates.append(
            {
                "first": layer.first.astype(np.int32).tolist(),
                "second": layer.second.astype(np.int32).tolist(),
                "first_coeff": layer.first_coeff.astype(np.float32).tolist(),
                "second_coeff": layer.second_coeff.astype(np.float32).tolist(),
                "const": layer.const.astype(np.float32).tolist(),
                "product_coeff": layer.product_coeff.astype(np.float32).tolist(),
            }
        )
    return {"n": int(circuit.n), "d": int(circuit.d), "gates": gates}


def _collect_prediction_tensor(
    predictions: object,
    *,
    width: int,
    depth: int,
) -> NDArray[np.float32]:
    try:
        output_iter = iter(cast(Any, predictions))
    except TypeError as exc:
        raise ValueError("Estimator must return an iterator of depth-row outputs.") from exc

    rows: list[NDArray[np.float32]] = []
    for depth_index in range(depth):
        try:
            raw_row = next(output_iter)
        except StopIteration as exc:
            raise ValueError("Estimator must emit exactly max_depth rows.") from exc
        except Exception as exc:
            raise ValueError(
                f"Estimator stream failed while producing depth row {depth_index}: {exc}"
            ) from exc
        rows.append(validate_depth_row(raw_row, width=width, depth_index=depth_index))

    try:
        _extra = next(output_iter)
    except StopIteration:
        pass
    except Exception as exc:
        raise ValueError(f"Estimator stream failed after emitting max_depth rows: {exc}") from exc
    else:
        raise ValueError("Estimator emitted more than max_depth rows.")

    return np.stack(rows, axis=0).astype(np.float32)


class InProcessRunner:
    """Local runner that executes estimator methods in this Python process."""

    def __init__(self) -> None:
        self._estimator: BaseEstimator | None = None
        self._limits: ResourceLimits | None = None
        self._started = False

    def start(
        self,
        entrypoint: EstimatorEntrypoint,
        context: SetupContext,
        limits: ResourceLimits,
    ) -> None:
        self.close()
        self._limits = limits
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

    def predict(self, circuit: Circuit, budget: int) -> PredictOutcome:
        if not self._started or self._estimator is None or self._limits is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="RUNNER_NOT_STARTED",
                    message="Runner must be started before calling predict.",
                ),
            )

        start_wall = time.time()
        start_cpu = time.process_time()
        try:
            raw_predictions = self._estimator.predict(circuit, budget)
        except Exception as exc:
            elapsed = time.time() - start_wall
            cpu_elapsed = time.process_time() - start_cpu
            rss_bytes = _rss_bytes()
            peak_rss_bytes = max(_peak_rss_bytes(), rss_bytes)
            return PredictOutcome(
                predictions=None,
                wall_time_s=float(elapsed),
                cpu_time_s=float(cpu_elapsed),
                rss_bytes=int(rss_bytes),
                peak_rss_bytes=int(peak_rss_bytes),
                status="runtime_error",
                error_message=str(exc),
            )

        try:
            predictions = _collect_prediction_tensor(
                raw_predictions,
                width=circuit.n,
                depth=circuit.d,
            )
        except ValueError as exc:
            elapsed = time.time() - start_wall
            cpu_elapsed = time.process_time() - start_cpu
            rss_bytes = _rss_bytes()
            peak_rss_bytes = max(_peak_rss_bytes(), rss_bytes)
            return PredictOutcome(
                predictions=None,
                wall_time_s=float(elapsed),
                cpu_time_s=float(cpu_elapsed),
                rss_bytes=int(rss_bytes),
                peak_rss_bytes=int(peak_rss_bytes),
                status="protocol_error",
                error_message=str(exc),
            )

        elapsed = time.time() - start_wall
        cpu_elapsed = time.process_time() - start_cpu
        rss_bytes = _rss_bytes()
        peak_rss_bytes = max(_peak_rss_bytes(), rss_bytes)
        if elapsed > self._limits.predict_timeout_s:
            return PredictOutcome(
                predictions=None,
                wall_time_s=float(elapsed),
                cpu_time_s=float(cpu_elapsed),
                rss_bytes=int(rss_bytes),
                peak_rss_bytes=int(peak_rss_bytes),
                status="timeout",
                error_message=(
                    f"predict exceeded timeout ({elapsed:.6f}s > "
                    f"{self._limits.predict_timeout_s:.6f}s)"
                ),
            )
        return PredictOutcome(
            predictions=predictions,
            wall_time_s=float(elapsed),
            cpu_time_s=float(cpu_elapsed),
            rss_bytes=int(rss_bytes),
            peak_rss_bytes=int(peak_rss_bytes),
            status="ok",
            error_message=None,
        )

    def predict_batch(self, circuits: list[Circuit], budget: int) -> list[PredictOutcome]:
        return [self.predict(circuit, budget) for circuit in circuits]

    def close(self) -> None:
        if self._estimator is not None:
            teardown = getattr(self._estimator, "teardown", None)
            if callable(teardown):
                teardown()
        self._estimator = None
        self._limits = None
        self._started = False


class SubprocessRunner:
    """Runner that executes estimators in a dedicated subprocess worker."""

    def __init__(
        self,
        *,
        worker_command: list[str] | None = None,
    ) -> None:
        self._worker_command = (
            worker_command
            if worker_command is not None
            else [sys.executable, "-m", "circuit_estimation.subprocess_worker"]
        )
        self._process: subprocess.Popen[str] | None = None
        self._limits: ResourceLimits | None = None
        self._started = False

    def start(
        self,
        entrypoint: EstimatorEntrypoint,
        context: SetupContext,
        limits: ResourceLimits,
    ) -> None:
        self.close()
        self._limits = limits
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
                    "max_depth": context.max_depth,
                    "budgets": list(context.budgets),
                    "time_tolerance": context.time_tolerance,
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
                RunnerErrorDetail(
                    code="SETUP_TIMEOUT",
                    message="worker setup timed out waiting for startup response.",
                ),
            ) from exc
        except RunnerError as exc:
            stderr_tail = self._read_stderr_tail()
            detail_message = exc.detail.message
            if stderr_tail:
                detail_message = f"{detail_message} stderr: {stderr_tail}"
            self._terminate_process()
            raise RunnerError(
                "setup",
                RunnerErrorDetail(
                    code="SETUP_PROTOCOL_ERROR",
                    message=detail_message,
                ),
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

    def predict(self, circuit: Circuit, budget: int) -> PredictOutcome:
        if not self._started or self._process is None or self._limits is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="RUNNER_NOT_STARTED",
                    message="Runner must be started before calling predict.",
                ),
            )

        start_wall = time.time()
        start_cpu = time.process_time()
        try:
            self._send_request(
                {
                    "command": "predict",
                    "budget": int(budget),
                    "circuit": _circuit_to_payload(circuit),
                }
            )
            response = self._read_response(timeout_s=self._limits.predict_timeout_s)
        except TimeoutError:
            self._terminate_process()
            elapsed = time.time() - start_wall
            cpu_elapsed = time.process_time() - start_cpu
            rss_bytes = _rss_bytes()
            peak_rss_bytes = max(_peak_rss_bytes(), rss_bytes)
            self._started = False
            return PredictOutcome(
                predictions=None,
                wall_time_s=float(elapsed),
                cpu_time_s=float(cpu_elapsed),
                rss_bytes=int(rss_bytes),
                peak_rss_bytes=int(peak_rss_bytes),
                status="timeout",
                error_message="predict timed out waiting for worker response.",
            )
        except RunnerError as exc:
            elapsed = time.time() - start_wall
            cpu_elapsed = time.process_time() - start_cpu
            rss_bytes = _rss_bytes()
            peak_rss_bytes = max(_peak_rss_bytes(), rss_bytes)
            return PredictOutcome(
                predictions=None,
                wall_time_s=float(elapsed),
                cpu_time_s=float(cpu_elapsed),
                rss_bytes=int(rss_bytes),
                peak_rss_bytes=int(peak_rss_bytes),
                status="protocol_error",
                error_message=exc.detail.message,
            )

        elapsed = time.time() - start_wall
        cpu_elapsed = time.process_time() - start_cpu
        rss_bytes = _rss_bytes()
        peak_rss_bytes = max(_peak_rss_bytes(), rss_bytes)
        status = str(response.get("status", "protocol_error"))
        if status == "ok":
            raw_predictions = response.get("predictions")
            if not isinstance(raw_predictions, list):
                return PredictOutcome(
                    predictions=None,
                    wall_time_s=float(elapsed),
                    cpu_time_s=float(cpu_elapsed),
                    rss_bytes=int(rss_bytes),
                    peak_rss_bytes=int(peak_rss_bytes),
                    status="protocol_error",
                    error_message="Worker response missing predictions list.",
                )
            predictions = np.asarray(raw_predictions, dtype=np.float32)
            return PredictOutcome(
                predictions=predictions,
                wall_time_s=float(elapsed),
                cpu_time_s=float(cpu_elapsed),
                rss_bytes=int(rss_bytes),
                peak_rss_bytes=int(peak_rss_bytes),
                status="ok",
                error_message=None,
            )

        mapped_status: PredictStatus = (
            cast(PredictStatus, status)
            if status in {"timeout", "oom", "runtime_error", "protocol_error"}
            else "protocol_error"
        )
        return PredictOutcome(
            predictions=None,
            wall_time_s=float(elapsed),
            cpu_time_s=float(cpu_elapsed),
            rss_bytes=int(rss_bytes),
            peak_rss_bytes=int(peak_rss_bytes),
            status=mapped_status,
            error_message=str(response.get("error_message", "worker predict failed")),
        )

    def predict_batch(self, circuits: list[Circuit], budget: int) -> list[PredictOutcome]:
        return [self.predict(circuit, budget) for circuit in circuits]

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
        self._started = False

    def _send_request(self, payload: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="WORKER_IO_ERROR",
                    message="Worker stdin is unavailable.",
                ),
            )
        try:
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="WORKER_BROKEN_PIPE",
                    message="Worker process closed stdin unexpectedly.",
                ),
            ) from exc

    def _read_response(self, timeout_s: float) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="WORKER_IO_ERROR",
                    message="Worker stdout is unavailable.",
                ),
            )
        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(timeout=timeout_s)
            if not events:
                raise TimeoutError("worker response timed out")
            line = self._process.stdout.readline()
        finally:
            selector.close()
        if line == "":
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="WORKER_EOF",
                    message="Worker closed stdout unexpectedly.",
                ),
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="WORKER_PROTOCOL_ERROR",
                    message="Worker returned invalid JSON response.",
                    details={"raw": line.strip()},
                ),
            ) from exc
        if not isinstance(payload, dict):
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="WORKER_PROTOCOL_ERROR",
                    message="Worker response must be a JSON object.",
                ),
            )
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
        if not stderr:
            return ""
        lines = stderr.splitlines()
        return lines[-1]

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parents[1])
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src_root if not current else f"{src_root}{os.pathsep}{current}"
        return env
