"""Canonical compute-budget scoring primitives shared by whestbench and downstream evaluators.

Single source of truth for the budget math:
  effective compute   C = F + LAMBDA * R     (F = FLOPs, R = residual wall-time seconds)
  combined exhaustion C > B                  (strict; no grace margin)
  score multiplier    max(0.1, C / B)        (floored at 0.1, uncapped above; 1.0 on failure)

Kept import-light (no flopscope / datasets) so other packages can import it cheaply.
"""

from __future__ import annotations

# FLOP-equivalent rate for residual wall time (NeurIPS proposal: 1e11 FLOPs/second).
LAMBDA_FLOPS_PER_SECOND: float = 1e11


def effective_compute(
    flops_used: float,
    residual_wall_time_s: float,
    lambda_flops_per_second: float = LAMBDA_FLOPS_PER_SECOND,
) -> float:
    """C_m = F_m + lambda * R_m. ``lambda_flops_per_second`` defaults to the
    canonical rate; pass a value to re-calibrate without touching code."""
    return float(flops_used) + float(lambda_flops_per_second) * float(residual_wall_time_s)


def is_combined_budget_exhausted(
    flops_used: float,
    residual_wall_time_s: float,
    flop_budget: float,
    lambda_flops_per_second: float = LAMBDA_FLOPS_PER_SECOND,
) -> bool:
    """True when combined effective compute strictly exceeds the budget.

    Strict ``>`` (no grace margin): ``C_m == B_m`` is within budget. ``R_m`` is
    wall-clock and therefore noisy, so the boundary is a cliff — accepted by
    design; it only bites submissions intentionally maxing both FLOPs and
    residual time near 100%.
    """
    if flop_budget <= 0:
        return False
    return effective_compute(flops_used, residual_wall_time_s, lambda_flops_per_second) > float(
        flop_budget
    )


def score_multiplier(effective_compute: float, flop_budget: float, *, failed: bool) -> float:
    """Per-MLP multiplier: 1.0 on failure (or no budget), else ``max(0.1, C/B)`` — uncapped above."""
    if failed or flop_budget <= 0:
        return 1.0
    return max(0.1, float(effective_compute) / float(flop_budget))
