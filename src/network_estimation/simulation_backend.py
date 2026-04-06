"""Abstract base class for MLP forward pass backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import mechestim as me

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
    def run_mlp(self, mlp: MLP, inputs: me.ndarray) -> me.ndarray: ...

    @abstractmethod
    def run_mlp_all_layers(
        self, mlp: MLP, inputs: me.ndarray
    ) -> List[me.ndarray]: ...

    @abstractmethod
    def sample_layer_statistics(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[me.ndarray, me.ndarray, float]: ...
