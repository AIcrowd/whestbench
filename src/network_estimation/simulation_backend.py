"""Abstract base class for MLP forward pass backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP


class SimulationBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @classmethod
    def install_hint(cls) -> str:
        return ""

    @abstractmethod
    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]: ...

    @abstractmethod
    def run_mlp_all_layers(self, mlp: MLP, inputs: NDArray[np.float32]) -> List[NDArray[np.float32]]: ...

    @abstractmethod
    def output_stats(self, mlp: MLP, n_samples: int) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]: ...
