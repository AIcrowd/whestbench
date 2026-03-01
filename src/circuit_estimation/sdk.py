"""Participant-facing estimator SDK interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .domain import Circuit


@dataclass(frozen=True, slots=True)
class SetupContext:
    width: int
    max_depth: int
    budgets: tuple[int, ...]
    time_tolerance: float
    api_version: str
    scratch_dir: str | None = None


class BaseEstimator(ABC):
    def setup(self, context: SetupContext) -> None:
        """Optional one-time setup hook."""

    @abstractmethod
    def predict(self, circuit: Circuit, budget: int) -> np.ndarray:
        """Predict per-wire means for a single circuit and budget."""

    def predict_batch(self, circuits: Iterable[Circuit], budget: int) -> np.ndarray:
        """Default batch prediction via sequential single-circuit calls."""
        return np.stack([self.predict(circuit, budget) for circuit in circuits], axis=0)

    def teardown(self) -> None:
        """Optional one-time cleanup hook."""
