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
two modes is hard-coded to a phase: pass any rate you like.

Reproducing an older round means restoring ALL of its settings, not just the
rate — restoring only some re-scores the run under a mix of both rulebooks and
produces a number that matches neither. Every round is therefore kept whole, in
``ROUNDS``, keyed by its dataset tag:

    from whestbench.budget import ROUNDS
    r = ROUNDS["v1-phase1"]
    r.flop_budget, r.lambda_flops_per_second, r.residual_wall_time_limit_s,
    r.wall_time_limit_s, r.width, r.depth

The two settings easiest to forget are the wall cap and the gate. A submission
taking between 60 s and 120 s was time_exhausted under the v1-* rounds but passes
under the current 120 s default; and those rounds gated nothing, so leaving
today's 0.4 s residual cap in place fails MLPs they would have allowed.

``CURRENT_ROUND`` is the round being graded, and every default below is derived
from it, so advancing a phase is one edit rather than a hunt through the
codebase. See docs/reference/rounds.md for the round-by-round comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class RoundConfig:
    """Every setting that defines one competition round.

    Rounds are kept side by side rather than replaced, because re-scoring an
    older submission means restoring ALL of these together. Restoring only some
    of them scores that run under a mix of two rulebooks and silently produces a
    number that matches neither -- see the module docstring above.

    ``lambda_flops_per_second`` is the residual RATE. It is meaningful only in
    the priced model (``residual_mode == "priced"``); under gating the rate is
    0.0 and ``residual_wall_time_limit_s`` does the work instead.
    """

    #: Dataset revision tag on the HF repos, e.g. ``"v2-phase2"``.
    tag: str
    #: MLP shape the round was baked at.
    width: int
    depth: int
    #: Ground-truth Monte-Carlo draws per MLP.
    n_samples: int
    #: Per-MLP effective-compute budget B_m.
    flop_budget: int
    #: Residual rate. See ``residual_mode``.
    lambda_flops_per_second: float
    #: Hard cap on residual seconds, or ``None`` when the round gated nothing.
    residual_wall_time_limit_s: Optional[float]
    #: Per-``predict()`` wall-clock cap.
    wall_time_limit_s: float
    #: ``"priced"`` (residual converted to FLOPs via lambda) or ``"gated"``
    #: (residual capped separately and not priced).
    residual_mode: str
    #: One-line summary of what changed relative to the previous round.
    note: str


WARMUP_ROUND = RoundConfig(
    tag="v1-warmup",
    width=256,
    depth=8,
    n_samples=1_000_000_000,
    flop_budget=68_000_000_000,  # 6.8e10
    lambda_flops_per_second=1e11,
    residual_wall_time_limit_s=None,
    wall_time_limit_s=60.0,
    residual_mode="priced",
    note="First public round. Residual wall time priced at 1e11; nothing gated.",
)

PHASE1_ROUND = RoundConfig(
    tag="v1-phase1",
    width=256,
    depth=32,
    n_samples=1_000_000_000,
    flop_budget=272_000_000_000,  # 2.72e11
    lambda_flops_per_second=1e11,
    residual_wall_time_limit_s=None,
    wall_time_limit_s=60.0,
    residual_mode="priced",
    note="Deeper MLPs (8 -> 32) and a 4x budget. Same priced-residual rulebook.",
)

PHASE2_ROUND = RoundConfig(
    tag="v2-phase2",
    width=1024,
    depth=16,
    n_samples=1_000_000_000,
    flop_budget=2**41,  # 2,199,023,255,552
    lambda_flops_per_second=0.0,
    residual_wall_time_limit_s=0.4,
    wall_time_limit_s=120.0,
    residual_mode="gated",
    note=(
        "Wider and shallower (256x32 -> 1024x16). Residual PRICING is deprecated: "
        "lambda is 0.0 and residual time is capped at 0.4 s instead, so C == F and "
        "the FLOP budget means what it says."
    ),
)

#: Every round, newest last. Keyed by dataset tag so a metadata revision maps
#: straight onto the rulebook it was scored under.
ROUNDS: Dict[str, RoundConfig] = {r.tag: r for r in (WARMUP_ROUND, PHASE1_ROUND, PHASE2_ROUND)}

#: The round currently being graded. Every default below derives from it, so
#: advancing a phase is one edit here rather than a hunt through the codebase.
CURRENT_ROUND: RoundConfig = PHASE2_ROUND

# --- Derived defaults ---------------------------------------------------------
# These names predate RoundConfig and stay for compatibility, but they are now
# derived rather than restated so the two can never disagree.

DEFAULT_FLOP_BUDGET: int = CURRENT_ROUND.flop_budget
PHASE1_FLOP_BUDGET: int = PHASE1_ROUND.flop_budget
WARMUP_FLOP_BUDGET: int = WARMUP_ROUND.flop_budget

# Phase 1 rate: residual wall time priced at 1e11 FLOP-equivalents per second.
# Pass this explicitly to re-score a Phase 1 or warmup round.
PHASE1_LAMBDA_FLOPS_PER_SECOND: float = PHASE1_ROUND.lambda_flops_per_second

# The default. 0.0 means residual wall time is not priced into effective compute
# at all — it is gated by residual_wall_time_limit_s instead — so C == F.
DEFAULT_LAMBDA_FLOPS_PER_SECOND: float = CURRENT_ROUND.lambda_flops_per_second

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
