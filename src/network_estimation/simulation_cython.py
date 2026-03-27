"""Cython backend — compiled loop dispatch with buffer reuse."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .simulation_backend import SimulationBackend

try:
    from . import _cython_kernels

    _HAS_CYTHON_EXT = True
except ImportError:
    _HAS_CYTHON_EXT = False


def _pick_chunk_size(width: int) -> int:
    return max(1024, min(16384, 2**20 // width))


class CythonBackend(SimulationBackend):
    @property
    def name(self) -> str:
        return "cython"

    @classmethod
    def is_available(cls) -> bool:
        return _HAS_CYTHON_EXT

    @classmethod
    def install_hint(cls) -> str:
        return "pip install cython>=3.0 && python setup_cython.py build_ext --inplace"

    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        result = _cython_kernels.forward_pass(inputs, mlp.weights)
        return np.asarray(result, dtype=np.float32)

    def run_mlp_matmul_only(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Matmul-only forward pass using numpy (no Cython kernel for matmul-only yet)."""
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        for w in mlp.weights:
            x = x @ w
        return x

    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        layers = _cython_kernels.forward_pass_all_layers(inputs, mlp.weights)
        return [np.asarray(layer, dtype=np.float32) for layer in layers]

    def sample_layer_statistics(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)
        layer_sums = np.zeros((depth, width), dtype=np.float64)
        final_sum_sq = np.zeros(width, dtype=np.float64)
        n_processed = 0
        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = np.random.default_rng().standard_normal((n, width), dtype=np.float32)
            layers = _cython_kernels.forward_pass_all_layers(x, mlp.weights)
            for layer_idx, layer_out in enumerate(layers):
                layer_sums[layer_idx] += layer_out.sum(axis=0).astype(np.float64)
            final_out = layers[-1]
            final_sum_sq += (final_out.astype(np.float64) ** 2).sum(axis=0)
            n_processed += n
        layer_means = (layer_sums / n_processed).astype(np.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(
            np.mean(final_sum_sq / n_processed - final_mean.astype(np.float64) ** 2)
        )
        return layer_means, final_mean, avg_variance
