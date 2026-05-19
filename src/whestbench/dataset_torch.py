"""Torch-backed variant of create_dataset for GPU acceleration on large bakes.

This module is a power-user drop-in alternative to whestbench.dataset.create_dataset
for n_samples >= 10^8 scenarios where the flopscope CPU path is too slow.
Torch is an optional dependency: install via `pip install whestbench[gpu]`.
"""

from __future__ import annotations

import math
from typing import Any, Dict


def _synthesize_sampling_breakdown(
    *,
    width: int,
    depth: int,
    n_samples: int,
    wall_time_s: float,
    flop_budget: int,
) -> Dict[str, Any]:
    """Closed-form analog of flopscope's BudgetContext.summary_dict for the torch path.

    The torch path computes outside flopscope's instrumentation, so this helper
    synthesizes the same dict shape using analytical FLOP counts. Verified
    against flopscope's actual count by test_closed_form_matches_flopscope_count.

    Output shape mirrors flopscope's normalized output exactly:
    - Top-level keys: flop_budget, flops_used, flops_remaining, wall_time_s,
      flopscope_backend_time_s, flopscope_overhead_time_s, residual_wall_time_s,
      by_namespace.
    - by_namespace is a FLAT dict keyed by dot-notation strings (e.g.
      "sampling.sample_layer_statistics"), NOT nested dicts.

    Formula derivation (matched exactly against flopscope's operation-level accounting):

    Per-sample ops (scale with n_samples):
      - standard_normal((n, width)):  16 FLOPs/element * n * width
        (flopscope counts RNG generation at 16 FLOPs/element, not 1)
      - array cast to fnp:             1 FLOPs/element * n * width
      - matmul (n,w)@(w,w) per layer:  n * w^2 per layer
      - maximum (ReLU) per layer:      n * w per layer      (1 FLOP/element)
      - sum along axis=0 per layer:    n * w per layer      (input size, per flopscope docs)
      - power x_f64**2:               16 FLOPs/element * n * width
        (flopscope charges power at 16 FLOPs/element, unlike x*x which is 1)
      - sum for final_sum_sq:          n * width            (input size)

    Per-chunk ops (scale with n_chunks):
      - add layer_sums += per layer:   width per layer per chunk
      - add final_sum_sq += :          width per chunk

    Post-loop ops (once):
      - true_divide layer_sums/n:      depth * width
      - true_divide final_sum_sq/n:    width
      - power final_mean**2:           16 * width           (same 16 FLOPs/element rule)
      - subtract (sq/n - mean^2):      width
      - mean (avg variance):           width
    """
    # chunk_size mirrors simulation._pick_chunk_size
    chunk_size = max(1024, min(16384, 2**20 // width))
    n_chunks = math.ceil(n_samples / chunk_size)

    # Per-sample costs
    standard_normal = 16 * n_samples * width
    array_cast = n_samples * width
    matmul = depth * n_samples * width * width
    relu = depth * n_samples * width
    sum_layer = depth * n_samples * width
    power_sq = 16 * n_samples * width
    sum_sq = n_samples * width

    # Per-chunk costs
    add_costs = n_chunks * (depth * width + width)

    # Post-loop costs
    true_divide_layer = depth * width
    true_divide_sq = width
    power_mean = 16 * width
    subtract = width
    mean_reduction = width

    total = (
        standard_normal
        + array_cast
        + matmul
        + relu
        + sum_layer
        + power_sq
        + sum_sq
        + add_costs
        + true_divide_layer
        + true_divide_sq
        + power_mean
        + subtract
        + mean_reduction
    )

    flops_remaining = max(0, flop_budget - total)

    return {
        "flop_budget": flop_budget,
        "flops_used": total,
        "flops_remaining": flops_remaining,
        "wall_time_s": wall_time_s,
        "flopscope_backend_time_s": 0.0,
        "flopscope_overhead_time_s": 0.0,
        "residual_wall_time_s": wall_time_s,
        "by_namespace": {
            "sampling.sample_layer_statistics": {
                "flops_used": total,
                "calls": 0,
                "flopscope_backend_time_s": 0.0,
                "flopscope_overhead_time_s": 0.0,
                "operations": {},
            }
        },
    }


def _resolve_device(device: str) -> str:
    """Resolve a user-facing device string to a concrete torch device kind.

    Args:
        device: One of "auto", "cuda", "mps", "cpu".

    Returns:
        A concrete device kind: "cuda", "mps", or "cpu". Never "auto".

    Raises:
        ValueError: If device is not one of the accepted values.
        RuntimeError: If an explicit device is requested but unavailable.
    """
    import torch  # local import: torch is an optional dep

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Either CUDA is not installed, or torch was built without "
                "CUDA support. For dev without a GPU, use device='cpu'. "
                "Install: pip install whestbench[gpu]"
            )
        return "cuda"
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested but torch.backends.mps.is_available() is False. "
                "MPS is only supported on Apple Silicon with macOS 12.3+. "
                "For dev elsewhere, use device='cpu'."
            )
        return "mps"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"device must be one of 'auto', 'cuda', 'mps', 'cpu'; got {device!r}")


def _auto_mlps_per_batch(*, n_mlps: int) -> int:
    """Default mlps_per_batch: cap at 16 to bound GPU memory growth."""
    return min(n_mlps, 16)


def _auto_chunk_size(*, device: str, width: int, mlps_per_batch: int) -> int:
    """Default chunk_size.

    On cuda: targets ~25% of free GPU memory for the activations tensor,
    clamped to [65536, 1<<20]. On mps/cpu: fixed 65536 (good balance of
    kernel-launch amortization and memory).
    """
    if device != "cuda":
        return 65536
    import torch  # local

    free_bytes, _ = torch.cuda.mem_get_info()
    target_bytes = min(2 * 1024**3, free_bytes // 4)
    size = target_bytes // (mlps_per_batch * width * 4)
    return max(65536, min(1 << 20, int(size)))
