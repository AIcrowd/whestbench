# pyright: reportMissingImports=false
"""Torch port of sample_layer_statistics for GPU/CPU torch execution.

Mirrors src/whestbench/simulation.py:sample_layer_statistics but operates on
a batch of MLPs in parallel via torch.bmm. Precision strategy matches CPU:
fp32 matmul + fp64 accumulators to avoid catastrophic accumulation error at
large N.

Precision strategy:
    fp32 matmul + fp64 accumulators on cpu/cuda for numerical stability at
    large N. On MPS (which doesn't support fp64), fp32 accumulators are used
    instead — this is acceptable for dev workflows (N ≤ 10^5) where fp32
    error is comparable to MC noise. For production N=10^9 bakes, use
    device='cuda'.
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
        avg_variances: tensor of shape (B,) — mean per-neuron variance at the
            final layer. float64 on cpu/cuda; float32 on mps (fp64 unsupported
            there). Callers should cast to float64 after moving to CPU if a
            consistent dtype is required.
    """
    B, depth, width, _ = weights_batch.shape
    device = weights_batch.device

    # MPS doesn't support fp64 — fall back to fp32 accumulators on MPS.
    # On CUDA/CPU we keep fp64 to avoid catastrophic accumulation at large N.
    # MPS is intended for dev work where N is small (≤ 10^5), so fp32 error
    # (~O(N * eps_fp32) ≈ 0.01 at N=10^5) is comparable to MC noise and acceptable.
    accum_dtype = torch.float32 if device.type == "mps" else torch.float64

    layer_sums = torch.zeros((B, depth, width), dtype=accum_dtype, device=device)
    final_sum_sq = torch.zeros((B, width), dtype=accum_dtype, device=device)
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
            layer_sums[:, layer_idx] += x.to(accum_dtype).sum(dim=1)

        final_sum_sq += (x.to(accum_dtype) ** 2).sum(dim=1)
        n_processed += n

        if progress is not None:
            progress({"completed": chunk_index + 1, "total": n_chunks, "unit": "chunks"})

    layer_means = (layer_sums / n_processed).to(torch.float32)
    final_means = layer_means[:, -1].contiguous()
    final_means_acc = final_means.to(accum_dtype)
    avg_variances = (final_sum_sq / n_processed - final_means_acc**2).mean(dim=1)

    return layer_means, final_means, avg_variances
