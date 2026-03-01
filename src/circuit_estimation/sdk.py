"""Participant-facing estimator base interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit


class BaseEstimator(ABC):
    """Streaming estimator contract for participant implementations."""

    @abstractmethod
    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        """Yield one prediction vector per depth for ``circuit``."""
        raise NotImplementedError

    def setup(self, context: object) -> None:
        """Optional one-time setup hook before prediction calls."""
        return None

    def teardown(self) -> None:
        """Optional cleanup hook after scoring completes."""
        return None
