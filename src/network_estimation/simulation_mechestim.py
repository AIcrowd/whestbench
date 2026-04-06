"""Mechestim backend — reference implementation using analytical FLOP counting."""

from __future__ import annotations

from typing import List, Tuple

import mechestim as me

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

        layer_sums = me.zeros((depth, width), dtype=me.float64)
        final_sum_sq = me.zeros(width, dtype=me.float64)
        n_processed = 0

        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = me.array(me.random.default_rng().standard_normal((n, width)).astype(me.float32))
            for layer_idx, w in enumerate(mlp.weights):
                x = me.maximum(me.matmul(x, w), 0.0)
                x_f64 = me.asarray(x, dtype=me.float64)
                layer_sums[layer_idx] += me.sum(x_f64, axis=0)
            x_f64 = me.asarray(x, dtype=me.float64)
            final_sum_sq += me.sum(x_f64**2, axis=0)
            n_processed += n

        layer_means = me.asarray(layer_sums / n_processed, dtype=me.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(
            me.mean(final_sum_sq / n_processed - me.asarray(final_mean, dtype=me.float64) ** 2)
        )
        return layer_means, final_mean, avg_variance
