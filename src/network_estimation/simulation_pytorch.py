"""PyTorch CPU backend for MLP forward pass simulation.

Uses PyTorch CPU BLAS (MKL/oneDNN) for matmul + fused ReLU, with weight
tensor caching and chunked streaming for sample_layer_statistics.
"""

from __future__ import annotations

import os
import weakref
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .simulation_backend import SimulationBackend

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is not installed")

# --- Thread control ---
# Respect NESTIM_MAX_THREADS / OMP_NUM_THREADS if set (e.g. by --max-threads),
# otherwise use all available CPU cores.
_env_limit = os.environ.get("NESTIM_MAX_THREADS") or os.environ.get("OMP_NUM_THREADS")
if _env_limit is not None:
    _n_threads = int(_env_limit)
else:
    _n_threads = os.cpu_count() or 1
torch.set_num_threads(_n_threads)

# --- Module-level weight cache ---
_weight_cache: Dict[int, list] = {}
_weak_refs: Dict[int, weakref.ref] = {}


def _get_torch_weights(mlp: MLP) -> list:
    """Get or create cached torch tensors for an MLP's weight matrices."""
    key = id(mlp)
    cached = _weight_cache.get(key)
    if cached is not None:
        return cached
    tensors = [torch.from_numpy(w) for w in mlp.weights]
    _weight_cache[key] = tensors

    def _on_finalize(ref: weakref.ref, k: int = key) -> None:
        _weight_cache.pop(k, None)
        _weak_refs.pop(k, None)

    _weak_refs[key] = weakref.ref(mlp, _on_finalize)
    return tensors


def _pick_chunk_size(width: int) -> int:
    """Choose chunk size targeting a 2-8 MB working set for L2/L3 cache."""
    return max(1024, min(16384, 2**20 // width))


class PyTorchBackend(SimulationBackend):
    """Simulation backend using PyTorch CPU tensors."""

    @property
    def name(self) -> str:
        return "pytorch"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch as _torch  # noqa: F811
            return True
        except ImportError:
            return False

    @classmethod
    def install_hint(cls) -> str:
        return "pip install torch>=2.0"

    @torch.no_grad()
    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass returning final-layer activations.

        Args:
            mlp: MLP to execute.
            inputs: Input matrix of shape ``(samples, mlp.width)``.

        Returns:
            Activations of shape ``(samples, mlp.width)`` after the last layer.
        """
        x = torch.from_numpy(inputs)
        for w in _get_torch_weights(mlp):
            x = torch.relu(x @ w)
        return x.numpy()

    @torch.no_grad()
    def run_mlp_matmul_only(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        x = torch.from_numpy(inputs)
        for w in _get_torch_weights(mlp):
            x = x @ w
        return x.numpy()

    @torch.no_grad()
    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        """Forward pass returning activations after each layer.

        Args:
            mlp: MLP to execute.
            inputs: Input matrix of shape ``(samples, mlp.width)``.

        Returns:
            List of ``depth`` arrays, each shape ``(samples, mlp.width)``.
        """
        x = torch.from_numpy(inputs)
        layers: List[NDArray[np.float32]] = []
        for w in _get_torch_weights(mlp):
            x = torch.relu(x @ w)
            layers.append(x.numpy())
        return layers

    @torch.no_grad()
    def sample_layer_statistics(
        self,
        mlp: MLP,
        n_samples: int,
        chunk_size: Optional[int] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """Compute per-layer means and average variance of the final layer.

        Uses chunked streaming to keep memory usage at O(chunk_size * width)
        instead of O(n_samples * width * depth).

        Args:
            mlp: MLP to evaluate.
            n_samples: Number of random Gaussian N(0,1) input vectors.
            chunk_size: Optional override for chunk size (for benchmarking).

        Returns:
            all_layer_means: shape ``(depth, width)`` -- mean activations per layer.
            final_mean: shape ``(width,)`` -- mean activations at the final layer.
            avg_variance: scalar -- average per-neuron variance at the final layer.
        """
        weights = _get_torch_weights(mlp)
        width = mlp.width
        depth = mlp.depth
        if chunk_size is None:
            chunk_size = _pick_chunk_size(width)

        # Online accumulators
        layer_sums = torch.zeros(depth, width, dtype=torch.float32)
        final_sum_sq = torch.zeros(width, dtype=torch.float32)

        n_processed = 0
        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = torch.randn(n, width, dtype=torch.float32)

            for layer_idx, w in enumerate(weights):
                x = torch.relu(x @ w)
                layer_sums[layer_idx] += x.sum(dim=0)

            final_sum_sq += (x * x).sum(dim=0)
            n_processed += n

        # Compute final statistics
        layer_means = (layer_sums / n_processed).numpy().astype(np.float32)
        final_mean = layer_means[-1].copy()
        final_mean_t = torch.from_numpy(final_mean)
        avg_variance = float(
            (final_sum_sq / n_processed - final_mean_t * final_mean_t).mean()
        )
        return layer_means, final_mean, avg_variance
