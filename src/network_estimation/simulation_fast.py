"""Optimized MLP forward pass using PyTorch CPU backend.

Drop-in replacement for ``simulation.py`` with identical API. Falls back
to the reference NumPy implementation when PyTorch is not installed.

Key optimizations:
- PyTorch CPU BLAS (MKL/oneDNN) for matmul + fused ReLU
- Weight tensor caching keyed on id(mlp) with weakref cleanup
- Chunked streaming in output_stats (O(MB) memory instead of O(GB))
"""

from __future__ import annotations

import os
import weakref
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if not _HAS_TORCH:
    from .simulation import output_stats, relu, run_mlp, run_mlp_all_layers
else:
    # Configure thread count at import time — cap at 4 to match target
    # 4-vCPU AWS compute-optimized instance.
    _MAX_THREADS = 4
    _n_threads = min(os.cpu_count() or _MAX_THREADS, _MAX_THREADS)
    torch.set_num_threads(_n_threads)

    # Weight cache: id(mlp) -> list of torch tensors.
    # MLP has frozen=True but contains a list field (unhashable), so we
    # cannot use WeakKeyDictionary. Instead we key on id() and register
    # a weak reference destructor to evict stale entries.
    _weight_cache: Dict[int, List[torch.Tensor]] = {}
    _weak_refs: Dict[int, weakref.ref[MLP]] = {}

    def _get_torch_weights(mlp: MLP) -> List[torch.Tensor]:
        """Get or create cached torch tensors for an MLP's weight matrices."""
        key = id(mlp)
        cached = _weight_cache.get(key)
        if cached is not None:
            return cached
        tensors = [torch.from_numpy(w) for w in mlp.weights]
        _weight_cache[key] = tensors

        def _on_finalize(ref: weakref.ref[MLP], k: int = key) -> None:
            _weight_cache.pop(k, None)
            _weak_refs.pop(k, None)

        _weak_refs[key] = weakref.ref(mlp, _on_finalize)
        return tensors

    def _pick_chunk_size(width: int) -> int:
        """Choose chunk size targeting a 2-8 MB working set for L2/L3 cache."""
        return max(1024, min(16384, 2**20 // width))

    def relu(x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Element-wise ReLU activation."""
        return np.maximum(x, np.float32(0.0))

    @torch.no_grad()
    def run_mlp(mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
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
    def run_mlp_all_layers(
        mlp: MLP, inputs: NDArray[np.float32]
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
    def output_stats(
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
                Defaults to an auto-tuned value based on width.

        Returns:
            all_layer_means: shape ``(depth, width)`` — mean activations per layer.
            final_mean: shape ``(width,)`` — mean activations at the final layer.
            avg_variance: scalar — average per-neuron variance at the final layer.
        """
        weights = _get_torch_weights(mlp)
        width = mlp.width
        depth = mlp.depth
        if chunk_size is None:
            chunk_size = _pick_chunk_size(width)

        # Online accumulators — only (depth, width) and (width,) sized
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
