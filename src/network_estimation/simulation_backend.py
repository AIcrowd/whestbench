"""Abstract base class for MLP forward pass backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP


@dataclass
class PrimitiveBreakdown:
    """Matmul vs ReLU timing breakdown derived by subtraction.

    Instead of instrumenting each layer with timers (which inflates times
    and breaks expression fusion), we time two tight loops:

    - ``run_mlp`` (fused matmul + relu)
    - ``run_mlp_matmul_only`` (matmul without relu)

    ReLU time is derived as ``fused_total - matmul_total``.

    Attributes:
        matmul_total: Median wall-clock seconds for the matmul-only forward pass.
        relu_total: Derived ReLU time (fused_total - matmul_total, clamped >= 0).
        fused_total: Median wall-clock seconds for the full fused forward pass.
    """

    matmul_total: float = 0.0
    relu_total: float = 0.0
    fused_total: float = 0.0

    @property
    def matmul_pct(self) -> float:
        return (self.matmul_total / self.fused_total * 100) if self.fused_total > 0 else 0.0

    @property
    def relu_pct(self) -> float:
        return (self.relu_total / self.fused_total * 100) if self.fused_total > 0 else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "matmul_total": self.matmul_total,
            "relu_total": self.relu_total,
            "fused_total": self.fused_total,
            "matmul_pct": round(self.matmul_pct, 1),
            "relu_pct": round(self.relu_pct, 1),
        }


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
    def run_mlp_matmul_only(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass with matmul only (no ReLU activation).

        Same loop structure as ``run_mlp`` but skips the activation function.
        Used by the profiler to estimate matmul vs ReLU time by subtraction.
        """
        ...

    @abstractmethod
    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]: ...

    @abstractmethod
    def sample_layer_statistics(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]: ...
