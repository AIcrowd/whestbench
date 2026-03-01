"""Runner interfaces and structured execution outcomes for estimator isolation."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit
from .loader import load_estimator_from_path
from .sdk import BaseEstimator, SetupContext

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
        estimator, _ = load_estimator_from_path(entrypoint.file_path, class_name=entrypoint.class_name)
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

        if not isinstance(raw_predictions, np.ndarray):
            return PredictOutcome(
                predictions=None,
                wall_time_s=float(elapsed),
                cpu_time_s=float(cpu_elapsed),
                rss_bytes=int(rss_bytes),
                peak_rss_bytes=int(peak_rss_bytes),
                status="protocol_error",
                error_message="Estimator predict must return numpy.ndarray.",
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
