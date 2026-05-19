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
) -> Dict[str, Any]:
    """Closed-form analog of flopscope's BudgetContext.summary_dict for the torch path.

    The torch path computes outside flopscope's instrumentation, so this helper
    synthesizes the same dict shape using analytical FLOP counts. Verified
    against flopscope's actual count by test_closed_form_matches_flopscope_count.

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

    return {
        "flops_used": total,
        "wall_time_s": wall_time_s,
        "flopscope_backend_time_s": 0.0,
        "flopscope_overhead_time_s": 0.0,
        "residual_wall_time_s": wall_time_s,
        "by_namespace": {
            "sampling": {
                "flops_used": total,
                "by_namespace": {
                    "sample_layer_statistics": {"flops_used": total},
                },
            }
        },
    }
