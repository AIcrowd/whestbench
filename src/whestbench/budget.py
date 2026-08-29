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
    #: Whether the flopscope this round was GRADED under bills float64 at 2x
    #: float32 and charges for the float32->float64 cast. False for rounds
    #: graded before flopscope v0.9.0, True after. Belongs here, beside the
    #: other rulebook settings, for the reason in the module docstring: it is
    #: one of the things you must restore to re-score an old round correctly.
    #: See :func:`mc_flops_per_sample`.
    dtype_aware_billing: bool
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
    # Predates the dtype-aware billing that landed in flopscope v0.9.0, so
    # float64 cost the same as float32 and the float32->float64 cast was free.
    dtype_aware_billing=False,
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
    dtype_aware_billing=True,
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
    dtype_aware_billing=True,
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


# --- The Monte-Carlo sampling reference (MC@B_m) -------------------------------
# "How good is a submission compared to just sampling?" is answered by MC@B_m:
# the score a pure Monte-Carlo estimator achieves if it spends the entire
# per-MLP budget on forward passes. It is published as `sampling_mse` on every
# graded submission and as the "vs Sampling" column on the leaderboard.
#
# The whole quantity reduces to sigma^2 / N, so it needs exactly two inputs: the
# MLPs' mean output variance (the dataset's own `avg_variance` column, baked at
# n_samples draws) and N, the number of samples the budget buys. N is the part
# that needs a cost model, and it is defined here so that every consumer --
# the grader, the challenge page, and the paper -- shares one definition.


def mc_flops_per_sample(width: int, depth: int, *, dtype_aware_billing: bool) -> int:
    """FLOPs flopscope charges for one Monte-Carlo sample through a (depth, width) MLP.

    One "sample" is one standard-normal input vector pushed through the network,
    with the per-layer activation means accumulated in float64 for numerical
    stability -- i.e. exactly what ``simulation.sample_layer_statistics`` does,
    minus the sum-of-squares it additionally needs for ``avg_variance``.

    Term by term, at flopscope's published prices::

        standard_normal((n, w))     16 * w          RNG, transcendental tier
        fnp.array(...) wrap              w          1 FLOP per element written
        matmul, once per layer      d * w * (2w-1)  w mults + (w-1) adds per output
        maximum(., 0.0), per layer  d * w           1 FLOP per element
        float64 accumulation        k * d * w       see below
        -----------------------------------------------------------------
        total                       2*d*w^2 + 17*w + k*d*w

    (the matmul's ``-d*w`` and the ReLU's ``+d*w`` cancel exactly, which is why
    the leading term is the clean ``2*d*w^2``.)

    **Why k depends on the round.** flopscope gained dtype-aware billing in
    **v0.9.0**: from that release a float64 operation costs 2x the
    same operation on float32, and the float32->float64 cast -- previously free
    -- costs 2 FLOPs per element. Before it, every dtype billed alike. The
    forward pass is float32 throughout and is unaffected (verified identical
    across every flopscope release from v0.2.0 to v0.12.1); only the float64
    accumulation moves, giving::

        k = 1   graded before flopscope v0.9.0   (asarray free, sum at rate 1)
        k = 4   graded from flopscope v0.9.0 on  (asarray 2/elem, sum at rate 2)

    Hence ``dtype_aware_billing`` is a per-round setting on
    :class:`RoundConfig` rather than a constant: it is one more thing you must
    restore to re-score an old round under its own rulebook.

    **This split is historical fidelity, not a claim about the physics.** For
    the final results in the paper we intend to re-score every round under the
    latest stable flopscope, at which point all rounds use ``k = 4`` and this
    parameter collapses to a constant.
    """
    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive.")
    k = 4 if dtype_aware_billing else 1
    rng = 16 * width
    wrap = width
    matmul = depth * width * (2 * width - 1)
    relu = depth * width
    accumulate = k * depth * width
    return rng + wrap + matmul + relu + accumulate


def mc_flops_per_sample_for_round(round_config: RoundConfig) -> int:
    """:func:`mc_flops_per_sample` resolved from a round's own geometry and regime."""
    return mc_flops_per_sample(
        round_config.width,
        round_config.depth,
        dtype_aware_billing=round_config.dtype_aware_billing,
    )


def mc_samples_at_budget(round_config: RoundConfig) -> float:
    """N -- how many Monte-Carlo samples the per-MLP budget B_m buys.

    Deliberately fractional, and fractional in every round: no budget divides
    evenly by its per-sample cost. Phase 2 looks as though it should -- its
    budget is 2**41 and the leading term of the per-sample cost, 2*d*w**2, is
    2**25 -- but the +17*w and +k*d*w terms break it, giving 33,637,376 per
    sample and N = 65,374.3995. The budget is a published rules figure and is
    used exactly as published; reading a round number into it has misled us
    before.
    """
    return round_config.flop_budget / mc_flops_per_sample_for_round(round_config)


def mc_at_bm(mean_avg_variance: float, round_config: RoundConfig) -> float:
    """MC@B_m -- the adjusted score full-budget Monte-Carlo sampling achieves.

    ``E[MSE(N)] = sigma^2 / N``, and MC spending the whole budget has FLOP
    multiplier ``max(0.1, B_m/B_m) = 1``, so the adjusted score is that MSE.

    ``mean_avg_variance`` MUST be the mean of the ``avg_variance`` column over
    **the same MLPs the result will be compared against** -- the graded split
    for a leaderboard, the convergence study's own MLPs for a convergence plot.
    Mixing populations is the mistake this signature is shaped to prevent: two
    samples of MLPs can carry materially different variance, so a reference
    borrowed from the wrong one mis-scales every comparison made against it.
    """
    if mean_avg_variance <= 0:
        raise ValueError("mean_avg_variance must be positive.")
    return mean_avg_variance / mc_samples_at_budget(round_config)


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
