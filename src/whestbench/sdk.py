"""Participant-facing estimator base interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import whest as we

from .domain import MLP


@dataclass(frozen=True)
class SetupContext:
    """Runtime context passed to ``BaseEstimator.setup``."""

    width: int
    depth: int
    flop_budget: int
    api_version: str
    scratch_dir: Optional[str] = None


class BaseEstimator(ABC):
    """Estimator contract for participant implementations.

    Participants subclass this and implement ``predict`` to return
    predicted means for all layers as a single ``(depth, width)`` whest array.
    """

    @abstractmethod
    def predict(self, mlp: MLP, budget: int) -> we.ndarray:
        """Return predicted means for all layers, shape ``(depth, width)``."""
        raise NotImplementedError

    def setup(self, context: SetupContext) -> None:
        """Optional one-time setup hook before prediction calls."""
        return None

    def teardown(self) -> None:
        """Optional cleanup hook after scoring completes."""
        return None
