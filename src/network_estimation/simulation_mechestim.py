"""Mechestim backend — reference implementation using analytical FLOP counting."""

from __future__ import annotations

from typing import List, Tuple

import mechestim as me
import numpy as np  # needed for internal accumulators (not FLOP-counted)

from .domain import MLP
from .simulation_backend import SimulationBackend


def _pick_chunk_size(width: int) -> int:
    return max(1024, min(16384, 2**20 // width))


class MechestimBackend(SimulationBackend):
    @property
    def name(self) -> str:
        return "mechestim"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def run_mlp(self, mlp: MLP, inputs: me.ndarray) -> me.ndarray:
        x = inputs
        for w in mlp.weights:
            x = me.maximum(me.matmul(x, w), 0.0)
        return x

    def run_mlp_all_layers(self, mlp: MLP, inputs: me.ndarray) -> List[me.ndarray]:
        x = inputs
        layers: List[me.ndarray] = []
        for w in mlp.weights:
            x = me.maximum(me.matmul(x, w), 0.0)
            layers.append(x)
        return layers

    def sample_layer_statistics(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[me.ndarray, me.ndarray, float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)

        # Accumulators — use numpy internally for sums, wrap at the end.
        layer_sums = np.zeros((depth, width), dtype=np.float64)
        final_sum_sq = np.zeros(width, dtype=np.float64)
        n_processed = 0

        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = me.array(me.random.default_rng().standard_normal((n, width)).astype(me.float32))
            for layer_idx, w in enumerate(mlp.weights):
                x = me.maximum(me.matmul(x, w), 0.0)
                # Extract to numpy for accumulation (not scored — this is ground truth)
                x_np = np.asarray(x, dtype=np.float64)
                layer_sums[layer_idx] += x_np.sum(axis=0)
            x_np = np.asarray(x, dtype=np.float64)
            final_sum_sq += (x_np**2).sum(axis=0)
            n_processed += n

        layer_means_np = (layer_sums / n_processed).astype(np.float32)
        final_mean_np = layer_means_np[-1].copy()
        avg_variance = float(
            np.mean(final_sum_sq / n_processed - final_mean_np.astype(np.float64) ** 2)
        )
        layer_means = me.array(layer_means_np)
        final_mean = me.array(final_mean_np)
        return layer_means, final_mean, avg_variance
