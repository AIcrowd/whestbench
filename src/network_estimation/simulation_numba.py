"""Numba JIT-compiled simulation backend for MLP forward passes.

Uses @njit with cache=True for persistent compilation. ReLU uses
np.maximum which Numba compiles to efficient SIMD code. Falls back
gracefully when numba is not installed.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .simulation_backend import SimulationBackend

try:
    from numba import njit

    _HAS_NUMBA = True
except (ImportError, OSError, SystemError):
    _HAS_NUMBA = False


def _pick_chunk_size(width: int) -> int:
    """Choose chunk size targeting a 2-8 MB working set for L2/L3 cache."""
    return max(1024, min(16384, 2**20 // width))


# ---------------------------------------------------------------------------
# JIT-compiled helpers (only defined when numba is available)
# ---------------------------------------------------------------------------
if _HAS_NUMBA:

    @njit(cache=True)
    def _relu_inplace(x: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        """Element-wise ReLU, modifying *x* in place."""
        np.maximum(x, np.float32(0.0), x)
        return x

    @njit(cache=True)
    def _forward_pass(
        inputs: np.ndarray,
        weights_tuple: tuple,  # type: ignore[type-arg]
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Forward pass returning only the final-layer activations."""
        x = inputs.copy()
        for k in range(len(weights_tuple)):
            x = x @ weights_tuple[k]
            _relu_inplace(x)
        return x

    @njit(cache=True)
    def _forward_pass_matmul_only(
        inputs: np.ndarray,
        weights_tuple: tuple,  # type: ignore[type-arg]
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Forward pass with matmul only (no ReLU)."""
        x = inputs.copy()
        for k in range(len(weights_tuple)):
            x = x @ weights_tuple[k]
        return x

    @njit(cache=True)
    def _forward_pass_all_layers(
        inputs: np.ndarray,
        weights_tuple: tuple,  # type: ignore[type-arg]
    ) -> tuple:  # type: ignore[type-arg]
        """Forward pass collecting activations after every layer.

        Returns a tuple of arrays (one per layer) to avoid Numba
        reflected-list issues.
        """
        n_layers = len(weights_tuple)
        x = inputs.copy()
        # Pre-allocate a single stacked buffer and slice later
        rows = inputs.shape[0]
        cols = inputs.shape[1]
        all_layers = np.empty((n_layers, rows, cols), dtype=np.float32)
        for k in range(n_layers):
            x = x @ weights_tuple[k]
            _relu_inplace(x)
            all_layers[k] = x
        return all_layers

    @njit(cache=True)
    def _forward_pass_layer_stats(
        inputs: np.ndarray,
        weights_tuple: tuple,  # type: ignore[type-arg]
    ) -> tuple:  # type: ignore[type-arg]
        """Forward pass that accumulates per-layer sums and final sum-of-squares.

        Returns (layer_sums_f64, final_sum_sq_f64).
        """
        n_layers = len(weights_tuple)
        rows, cols = inputs.shape
        layer_sums = np.zeros((n_layers, cols), dtype=np.float64)
        x = inputs.copy()
        for k in range(n_layers):
            x = x @ weights_tuple[k]
            _relu_inplace(x)
            for j in range(cols):
                s = 0.0
                for i in range(rows):
                    s += x[i, j]
                layer_sums[k, j] = s
        # Final layer sum of squares
        final_sum_sq = np.zeros(cols, dtype=np.float64)
        for j in range(cols):
            s = 0.0
            for i in range(rows):
                s += float(x[i, j]) * float(x[i, j])
            final_sum_sq[j] = s
        return layer_sums, final_sum_sq


# ---------------------------------------------------------------------------
# Public backend class
# ---------------------------------------------------------------------------
class NumbaBackend(SimulationBackend):
    """MLP simulation backend accelerated with Numba JIT compilation."""

    @property
    def name(self) -> str:
        return "numba"

    @classmethod
    def is_available(cls) -> bool:
        return _HAS_NUMBA

    @classmethod
    def install_hint(cls) -> str:
        return "pip install numba>=0.58"

    # -- forward pass -------------------------------------------------------

    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        weights_tuple = tuple(mlp.weights)
        return _forward_pass(inputs.astype(np.float32, copy=False), weights_tuple)

    def run_mlp_matmul_only(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        weights_tuple = tuple(mlp.weights)
        return _forward_pass_matmul_only(inputs.astype(np.float32, copy=False), weights_tuple)

    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        weights_tuple = tuple(mlp.weights)
        stacked = _forward_pass_all_layers(inputs.astype(np.float32, copy=False), weights_tuple)
        # stacked is (n_layers, samples, width) ndarray; split into list
        return [stacked[k] for k in range(stacked.shape[0])]

    # -- output statistics (chunked streaming) ------------------------------

    def sample_layer_statistics(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)
        weights_tuple = tuple(mlp.weights)

        layer_sums = np.zeros((depth, width), dtype=np.float64)
        final_sum_sq = np.zeros(width, dtype=np.float64)
        n_processed = 0

        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = np.random.default_rng().standard_normal((n, width), dtype=np.float32)
            chunk_sums, chunk_sq = _forward_pass_layer_stats(x, weights_tuple)
            layer_sums += chunk_sums
            final_sum_sq += chunk_sq
            n_processed += n

        layer_means = (layer_sums / n_processed).astype(np.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(
            np.mean(final_sum_sq / n_processed - final_mean.astype(np.float64) ** 2)
        )
        return layer_means, final_mean, avg_variance
