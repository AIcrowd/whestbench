"""Canonical compute-budget scoring primitives shared by whestbench and downstream evaluators.

Single source of truth for the budget math:
  effective compute   C = F + LAMBDA * R     (F = FLOPs, R = residual wall-time seconds)
  combined exhaustion C > B                  (strict; no grace margin)
  score multiplier    max(0.1, C / B)        (floored at 0.1, uncapped above; 1.0 on failure)

Kept import-light (no flopscope / datasets) so other packages can import it cheaply.

Residual wall time: gated, not priced
-------------------------------------
LAMBDA is the FLOP-equivalent price of one second of residual wall time — the
part of predict() that flopscope does not meter (participant Python, control
flow, GC). There are two ways to keep that from becoming a free lunch, and the
competition has used each in turn:

  PRICED (lambda > 0)   Residual seconds are converted to FLOPs and added to the
                        bill. Spending wall time is allowed but costs budget, so
                        C exceeds F and the two resources trade against each
                        other. This is the Phase 1 design, at 1e11 FLOPs/second.

  GATED (lambda == 0)   Residual time is not priced at all; it is capped
                        separately (ResourceLimits.residual_wall_time_limit_s,
                        0.4 s by default) and crossing the cap fails the MLP
                        outright. C is then exactly F, so the FLOP budget means
                        what it says. This is the Phase 2 design and the default
                        here.

The default is GATED, so C == F unless a caller opts back in. Nothing about the
two modes is hard-coded to a phase: pass any rate you like. To reproduce a Phase
1 round, pass PHASE1_LAMBDA_FLOPS_PER_SECOND (or --lambda-flops-per-second 1e11)
along with that round's flop_budget, and set residual_wall_time_limit_s=None so
the Phase 2 gate does not fire on a run that was never scored against one.
"""

from __future__ import annotations

# Phase 1 rate: residual wall time priced at 1e11 FLOP-equivalents per second.
# Pass this explicitly to re-score a Phase 1 round.
PHASE1_LAMBDA_FLOPS_PER_SECOND: float = 1e11

# The default. 0.0 means residual wall time is not priced into effective compute
# at all — it is gated by residual_wall_time_limit_s instead — so C == F.
DEFAULT_LAMBDA_FLOPS_PER_SECOND: float = 0.0

# Deprecated alias, kept so existing imports neither break nor silently change
# value. It has always meant the Phase 1 rate and still does; it is NOT the
# current default. Prefer the two explicit names above.
LAMBDA_FLOPS_PER_SECOND: float = PHASE1_LAMBDA_FLOPS_PER_SECOND


def effective_compute(
    flops_used: float,
    residual_wall_time_s: float,
    lambda_flops_per_second: float = DEFAULT_LAMBDA_FLOPS_PER_SECOND,
) -> float:
    """C_m = F_m + lambda * R_m.

    ``lambda_flops_per_second`` defaults to
    :data:`DEFAULT_LAMBDA_FLOPS_PER_SECOND` (0.0), which makes this return
    ``flops_used`` unchanged — residual wall time is gated by
    ``residual_wall_time_limit_s`` rather than priced. Pass
    :data:`PHASE1_LAMBDA_FLOPS_PER_SECOND` to re-score a Phase 1 round, or any
    other rate to re-calibrate without touching code.
    """
    return float(flops_used) + float(lambda_flops_per_second) * float(residual_wall_time_s)


def is_combined_budget_exhausted(
    flops_used: float,
    residual_wall_time_s: float,
    flop_budget: float,
    lambda_flops_per_second: float = DEFAULT_LAMBDA_FLOPS_PER_SECOND,
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
