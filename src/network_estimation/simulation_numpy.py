"""NumPy backend — reference implementation wrapping simulation.py logic."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from .domain import MLP
from .simulation_backend import SimulationBackend


def _pick_chunk_size(width: int) -> int:
    return max(1024, min(16384, 2**20 // width))


class NumPyBackend(SimulationBackend):
    @property
    def name(self) -> str:
        return "numpy"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        x = inputs
        for w in mlp.weights:
            x = np.maximum(x @ w, np.float32(0.0))
        return x

    def run_mlp_matmul_only(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        x = inputs
        for w in mlp.weights:
            x = x @ w
        return x

    def run_mlp_all_layers(self, mlp: MLP, inputs: NDArray[np.float32]) -> List[NDArray[np.float32]]:
        x = inputs
        layers: List[NDArray[np.float32]] = []
        for w in mlp.weights:
            x = np.maximum(x @ w, np.float32(0.0))
            layers.append(x)
        return layers

    def sample_layer_statistics(self, mlp: MLP, n_samples: int) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)
        layer_sums = np.zeros((depth, width), dtype=np.float64)
        final_sum_sq = np.zeros(width, dtype=np.float64)
        n_processed = 0
        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = np.random.randn(n, width).astype(np.float32)
            for layer_idx, w in enumerate(mlp.weights):
                x = np.maximum(x @ w, np.float32(0.0))
                layer_sums[layer_idx] += x.sum(axis=0).astype(np.float64)
            final_sum_sq += (x.astype(np.float64) ** 2).sum(axis=0)
            n_processed += n
        layer_means = (layer_sums / n_processed).astype(np.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(np.mean(final_sum_sq / n_processed - final_mean.astype(np.float64) ** 2))
        return layer_means, final_mean, avg_variance
