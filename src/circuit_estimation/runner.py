"""Runner interfaces and structured execution outcomes for estimator isolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit
from .sdk import SetupContext

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
