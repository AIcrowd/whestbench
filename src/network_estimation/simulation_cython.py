"""Cython backend — compiled matmul+ReLU with numpy Accelerate BLAS."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from .domain import MLP
from .simulation_backend import PrimitiveBreakdown, SimulationBackend

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

    def run_mlp_profiled(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], PrimitiveBreakdown]:
        import time

        breakdown = PrimitiveBreakdown()
        t_start = time.perf_counter()
        x = np.array(inputs, dtype=np.float32, copy=True)
        for w in mlp.weights:
            t0 = time.perf_counter()
            x = x @ w
            t1 = time.perf_counter()
            np.maximum(x, np.float32(0.0), out=x)
            t2 = time.perf_counter()
            breakdown.matmul.append(t1 - t0)
            breakdown.relu.append(t2 - t1)
        breakdown.total = time.perf_counter() - t_start
        breakdown.overhead = breakdown.total - breakdown.total_matmul - breakdown.total_relu
        return x, breakdown

    def run_mlp_all_layers(self, mlp: MLP, inputs: NDArray[np.float32]) -> List[NDArray[np.float32]]:
        layers = _cython_kernels.forward_pass_all_layers(inputs, mlp.weights)
        return [np.asarray(layer, dtype=np.float32) for layer in layers]

    def output_stats(self, mlp: MLP, n_samples: int) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)
        layer_sums = np.zeros((depth, width), dtype=np.float64)
        final_sum_sq = np.zeros(width, dtype=np.float64)
        n_processed = 0
        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = np.random.randn(n, width).astype(np.float32)
            layers = _cython_kernels.forward_pass_all_layers(x, mlp.weights)
            for layer_idx, layer_out in enumerate(layers):
                layer_sums[layer_idx] += layer_out.sum(axis=0).astype(np.float64)
            final_out = layers[-1]
            final_sum_sq += (final_out.astype(np.float64) ** 2).sum(axis=0)
            n_processed += n
        layer_means = (layer_sums / n_processed).astype(np.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(np.mean(final_sum_sq / n_processed - final_mean.astype(np.float64) ** 2))
        return layer_means, final_mean, avg_variance
