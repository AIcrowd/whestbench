"""Torch port of sample_layer_statistics for GPU/CPU torch execution.

Mirrors src/whestbench/simulation.py:sample_layer_statistics but operates on
a batch of MLPs in parallel via torch.bmm. Precision strategy matches CPU:
fp32 matmul + fp64 accumulators to avoid catastrophic accumulation error at
large N.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


def sample_layer_statistics_torch(
    weights_batch: torch.Tensor,
    n_samples: int,
    generators: List[torch.Generator],
    *,
    chunk_size: int,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched torch analog of sample_layer_statistics.

    Args:
        weights_batch: Float32 tensor of shape (B, depth, width, width) on device.
        n_samples: Number of i.i.d. N(0, I) input vectors to draw per MLP.
        generators: List of B torch.Generator objects (one per MLP in batch),
            each seeded independently.
        chunk_size: Number of samples per chunk (memory-bounded streaming).
        progress: Optional callback invoked once per processed chunk.

    Returns:
        layer_means: float32 tensor of shape (B, depth, width) — mean activation
            per neuron per layer per MLP.
        final_means: float32 tensor of shape (B, width) — last layer's means.
        avg_variances: float64 tensor of shape (B,) — mean per-neuron variance
            at the final layer.
    """
    B, depth, width, _ = weights_batch.shape
    device = weights_batch.device

    layer_sums = torch.zeros((B, depth, width), dtype=torch.float64, device=device)
    final_sum_sq = torch.zeros((B, width), dtype=torch.float64, device=device)
    n_processed = 0

    n_chunks = (n_samples + chunk_size - 1) // chunk_size

    x_buf = torch.empty((B, chunk_size, width), dtype=torch.float32, device=device)

    for chunk_index in range(n_chunks):
        start = chunk_index * chunk_size
        n = min(chunk_size, n_samples - start)

        if n == chunk_size:
            x = x_buf
        else:
            x = torch.empty((B, n, width), dtype=torch.float32, device=device)

        for b, gen in enumerate(generators):
            torch.randn((n, width), out=x[b], generator=gen)

        for layer_idx in range(depth):
            w = weights_batch[:, layer_idx]  # (B, width, width)
            x = torch.bmm(x, w).clamp_min_(0.0)
            layer_sums[:, layer_idx] += x.to(torch.float64).sum(dim=1)

        final_sum_sq += (x.to(torch.float64) ** 2).sum(dim=1)
        n_processed += n

        if progress is not None:
            progress({"completed": chunk_index + 1, "total": n_chunks, "unit": "chunks"})

    layer_means = (layer_sums / n_processed).to(torch.float32)
    final_means = layer_means[:, -1].contiguous()
    final_means_f64 = final_means.to(torch.float64)
    avg_variances = (final_sum_sq / n_processed - final_means_f64**2).mean(dim=1)

    return layer_means, final_means, avg_variances
