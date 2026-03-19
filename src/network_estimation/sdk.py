"""Participant-facing estimator base interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit


@dataclass(frozen=True, slots=True)
class SetupContext:
    """Runtime context passed to ``BaseEstimator.setup``.

    This keeps participant setup hooks self-contained and future-proof without
    requiring direct imports from scoring internals.
    """

    width: int
    max_depth: int
    budgets: tuple[int, ...]
    time_tolerance: float
    api_version: str
    scratch_dir: str | None = None


class BaseEstimator(ABC):
    """Streaming estimator contract for participant implementations."""

    @abstractmethod
    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        """Yield one prediction vector per depth for ``circuit``."""
        raise NotImplementedError

    def setup(self, context: SetupContext) -> None:
        """Optional one-time setup hook before prediction calls."""
        return None

    def teardown(self) -> None:
        """Optional cleanup hook after scoring completes."""
        return None
