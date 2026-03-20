"""Abstract base class for MLP forward pass backends."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP


@dataclass
class PrimitiveBreakdown:
    """Per-layer timing breakdown for matmul and ReLU primitives.

    Attributes:
        matmul: Wall-clock seconds for each layer's matrix multiply.
        relu: Wall-clock seconds for each layer's ReLU activation.
        overhead: Time not accounted for by matmul + relu (loop, copies, etc.).
        total: Wall-clock seconds for the entire forward pass.
    """

    matmul: List[float] = field(default_factory=list)
    relu: List[float] = field(default_factory=list)
    overhead: float = 0.0
    total: float = 0.0

    @property
    def total_matmul(self) -> float:
        return sum(self.matmul)

    @property
    def total_relu(self) -> float:
        return sum(self.relu)

    @property
    def matmul_pct(self) -> float:
        return (self.total_matmul / self.total * 100) if self.total > 0 else 0.0

    @property
    def relu_pct(self) -> float:
        return (self.total_relu / self.total * 100) if self.total > 0 else 0.0

    @property
    def overhead_pct(self) -> float:
        return (self.overhead / self.total * 100) if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "matmul_per_layer": self.matmul,
            "relu_per_layer": self.relu,
            "total_matmul": self.total_matmul,
            "total_relu": self.total_relu,
            "overhead": self.overhead,
            "total": self.total,
            "matmul_pct": round(self.matmul_pct, 1),
            "relu_pct": round(self.relu_pct, 1),
            "overhead_pct": round(self.overhead_pct, 1),
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
    def run_mlp_all_layers(self, mlp: MLP, inputs: NDArray[np.float32]) -> List[NDArray[np.float32]]: ...

    @abstractmethod
    def output_stats(self, mlp: MLP, n_samples: int) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]: ...

    def run_mlp_profiled(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], PrimitiveBreakdown]:
        """Forward pass with per-primitive timing breakdown.

        Default implementation uses NumPy matmul and np.maximum for ReLU.
        Backends should override to use their native primitives.
        """
        breakdown = PrimitiveBreakdown()
        t_start = time.perf_counter()
        x = inputs
        for w in mlp.weights:
            t0 = time.perf_counter()
            x = x @ w
            t1 = time.perf_counter()
            x = np.maximum(x, np.float32(0.0))
            t2 = time.perf_counter()
            breakdown.matmul.append(t1 - t0)
            breakdown.relu.append(t2 - t1)
        breakdown.total = time.perf_counter() - t_start
        breakdown.overhead = breakdown.total - breakdown.total_matmul - breakdown.total_relu
        return x, breakdown
