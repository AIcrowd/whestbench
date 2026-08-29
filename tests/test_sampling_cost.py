"""Gate on the Monte-Carlo per-sample cost model.

``budget.mc_flops_per_sample`` is a closed form for something flopscope actually
bills, so the property that matters is that it agrees with the accountant --
not that the arithmetic is self-consistent. These tests assert the closed form
equals flopscope's own MARGINAL charge for one extra sample, which is the same
gate ``test_torch_flop_synthesis.py`` applies to the bake's cost model.

The dtype-aware half (k=1 vs k=4) cannot be checked against a single installed
flopscope -- the two regimes are two different releases. What IS checkable here
is the k=4 branch against a current flopscope, plus the algebraic relationship
between the branches; the historical k=1 branch was verified against flopscope
0.2.0 through 0.8.0rc5 when the model was derived.
"""

from __future__ import annotations

import pytest

from whestbench.budget import (
    PHASE1_ROUND,
    PHASE2_ROUND,
    ROUNDS,
    WARMUP_ROUND,
    mc_at_bm,
    mc_flops_per_sample,
    mc_flops_per_sample_for_round,
    mc_samples_at_budget,
)

# (width, depth) for the three shipped rounds.
GEOMETRIES = [(256, 8), (256, 32), (1024, 16)]


def _marginal_flops_per_sample(width: int, depth: int) -> int:
    """What flopscope charges for one EXTRA sample, measured by differencing.

    Differencing rather than dividing: a single run also pays once-off costs
    (weight setup, the final stack) that do not scale with n, and the per-sample
    figure must exclude them.
    """
    flops = pytest.importorskip("flopscope")
    fnp = pytest.importorskip("flopscope.numpy")

    rng = fnp.random.default_rng(0)
    weights = [
        rng.standard_normal((width, width), dtype=fnp.float32) * ((2.0 / width) ** 0.5)
        for _ in range(depth)
    ]

    def used(n: int) -> int:
        with flops.budget(10**17, quiet=True) as ctx:
            x = fnp.array(fnp.random.default_rng(1).standard_normal((n, width), dtype=fnp.float32))
            for w in weights:
                x = fnp.maximum(fnp.matmul(x, w), 0.0)
                # float64 accumulation, for numerical stability -- the term the
                # dtype-aware billing change moves.
                fnp.sum(fnp.asarray(x, dtype=fnp.float64), axis=0)
            return int(ctx.flops_used)

    lo, hi = 8, 72
    return (used(hi) - used(lo)) // (hi - lo)


@pytest.mark.parametrize("width,depth", GEOMETRIES)
def test_matches_flopscope(width: int, depth: int) -> None:
    """The closed form must agree with the accountant, not just with itself."""
    measured = _marginal_flops_per_sample(width, depth)
    predicted = mc_flops_per_sample(width, depth, dtype_aware_billing=True)
    assert measured == predicted, (
        f"flopscope charges {measured:,} per sample at {width}x{depth}, "
        f"model predicts {predicted:,}"
    )


@pytest.mark.parametrize("width,depth", GEOMETRIES)
def test_closed_form_is_2dw2_plus_17w_plus_kdw(width: int, depth: int) -> None:
    """The documented algebra, spelled out independently of the implementation."""
    for k, dtype_aware in ((1, False), (4, True)):
        assert mc_flops_per_sample(width, depth, dtype_aware_billing=dtype_aware) == (
            2 * depth * width * width + 17 * width + k * depth * width
        )


def test_regimes_differ_only_by_the_float64_accumulation() -> None:
    """k=4 exceeds k=1 by exactly 3*d*w -- nothing else may move between them."""
    for width, depth in GEOMETRIES:
        pre = mc_flops_per_sample(width, depth, dtype_aware_billing=False)
        post = mc_flops_per_sample(width, depth, dtype_aware_billing=True)
        assert post - pre == 3 * depth * width


def test_round_regimes_are_the_ones_we_graded_under() -> None:
    """Pins each round's billing regime; see mc_flops_per_sample's docstring."""
    assert WARMUP_ROUND.dtype_aware_billing is False
    assert PHASE1_ROUND.dtype_aware_billing is True
    assert PHASE2_ROUND.dtype_aware_billing is True


def test_per_round_costs_are_the_published_values() -> None:
    """Regression lock on the three numbers every downstream consumer bakes in."""
    assert mc_flops_per_sample_for_round(WARMUP_ROUND) == 1_054_976
    assert mc_flops_per_sample_for_round(PHASE1_ROUND) == 4_231_424
    assert mc_flops_per_sample_for_round(PHASE2_ROUND) == 33_637_376


def test_samples_at_budget_is_fractional_and_not_rounded() -> None:
    """Phase 2 divides evenly; the other two must NOT be silently rounded to match."""
    assert mc_samples_at_budget(PHASE2_ROUND) == pytest.approx(65_374.4, abs=0.1)
    assert mc_samples_at_budget(PHASE1_ROUND) == pytest.approx(64_280.96, abs=0.01)
    assert mc_samples_at_budget(WARMUP_ROUND) == pytest.approx(64_456.44, abs=0.01)
    assert mc_samples_at_budget(WARMUP_ROUND) % 1 != 0


def test_mc_at_bm_is_sigma_squared_over_n() -> None:
    """The definition, stated independently of the implementation."""
    sigma_sq = 0.05
    for round_ in (WARMUP_ROUND, PHASE1_ROUND, PHASE2_ROUND):
        expected = sigma_sq * mc_flops_per_sample_for_round(round_) / round_.flop_budget
        assert mc_at_bm(sigma_sq, round_) == pytest.approx(expected, rel=1e-12)


def test_mc_at_bm_rescales_linearly_with_the_per_sample_cost() -> None:
    """A reference computed under a different per-sample cost differs by exactly
    the cost ratio.

    This is the property that lets a value derived under one cost model be
    reconciled against another without recomputing sigma^2 -- useful when
    comparing against a figure produced by a different accounting of the sampler.
    """
    sigma_sq = 0.05
    other_cost = mc_flops_per_sample_for_round(PHASE1_ROUND) * 1.01
    under_other = sigma_sq * other_cost / PHASE1_ROUND.flop_budget
    ratio = mc_flops_per_sample_for_round(PHASE1_ROUND) / other_cost
    assert mc_at_bm(sigma_sq, PHASE1_ROUND) == pytest.approx(under_other * ratio, rel=1e-12)


def test_every_round_has_a_regime() -> None:
    """A new round must state its regime rather than inheriting one by accident."""
    for tag, cfg in ROUNDS.items():
        assert isinstance(cfg.dtype_aware_billing, bool), tag
