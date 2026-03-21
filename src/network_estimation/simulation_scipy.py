"""SciPy BLAS backend for MLP forward pass.

Uses scipy.linalg.blas.sgemm for single-precision matrix multiplication,
which links to the system BLAS (OpenBLAS / MKL / Accelerate) without
requiring PyTorch or JAX.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .simulation_backend import SimulationBackend

try:
    from scipy.linalg.blas import sgemm as _sgemm

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _pick_chunk_size(width: int) -> int:
    """Choose chunk size targeting a 2-8 MB working set for L2/L3 cache."""
    return max(1024, min(16384, 2**20 // width))


class SciPyBackend(SimulationBackend):
    @property
    def name(self) -> str:
        return "scipy"

    @classmethod
    def is_available(cls) -> bool:
        try:
            from scipy.linalg.blas import sgemm  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def install_hint(cls) -> str:
        return "pip install scipy"

    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        for w in mlp.weights:
            x = _sgemm(1.0, x, w)
            np.maximum(x, 0.0, out=x)
        return x

    def run_mlp_matmul_only(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        for w in mlp.weights:
            x = _sgemm(1.0, x, w)
        return x

    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        layers: List[NDArray[np.float32]] = []
        for w in mlp.weights:
            x = _sgemm(1.0, x, w)
            np.maximum(x, 0.0, out=x)
            layers.append(x.copy())
        return layers

    def sample_layer_statistics(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)

        # Online accumulators in float64 for numerical stability
        layer_sums = np.zeros((depth, width), dtype=np.float64)
        final_sum_sq = np.zeros(width, dtype=np.float64)

        n_processed = 0
        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = np.random.randn(n, width).astype(np.float32)

            for layer_idx, w in enumerate(mlp.weights):
                x = _sgemm(1.0, x, w)
                np.maximum(x, 0.0, out=x)
                layer_sums[layer_idx] += x.sum(axis=0).astype(np.float64)

            final_sum_sq += (x.astype(np.float64) ** 2).sum(axis=0)
            n_processed += n

        layer_means = (layer_sums / n_processed).astype(np.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(
            np.mean(final_sum_sq / n_processed - final_mean.astype(np.float64) ** 2)
        )
        return layer_means, final_mean, avg_variance
