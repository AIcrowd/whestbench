from __future__ import annotations

import pytest

from whestbench.budget import (
    LAMBDA_FLOPS_PER_SECOND,
    effective_compute,
    is_combined_budget_exhausted,
    score_multiplier,
)


def test_lambda_constant():
    """The legacy name still means the Phase 1 rate and must not change value."""
    assert LAMBDA_FLOPS_PER_SECOND == 1e11


def test_effective_compute_adds_residual_at_lambda():
    # PRICED regime: 1e9 FLOPs + 1e11 * 0.5 s = 1e9 + 5e10 = 5.1e10.
    # The rate is named explicitly — the default is now 0 (gated), which would make
    # this a tautology rather than a test of the formula.
    assert effective_compute(1e9, 0.5, LAMBDA_FLOPS_PER_SECOND) == pytest.approx(5.1e10)


def test_effective_compute_default_is_gated_so_residual_is_free():
    """Default lambda is 0: residual seconds do not enter C at all, so C == F."""
    assert effective_compute(1e9, 0.5) == 1e9
    assert effective_compute(1e9, 1_000.0) == 1e9


def test_effective_compute_zero_residual_is_flops():
    assert effective_compute(1234.0, 0.0) == 1234.0


def test_combined_exhausted_is_strict_greater_than():
    budget = 1e10
    assert is_combined_budget_exhausted(1e10, 0.0, budget) is False  # C == B → not exhausted
    assert (
        is_combined_budget_exhausted(1e10, 1e-9, budget, LAMBDA_FLOPS_PER_SECOND) is True
    )  # C just over B, priced


def test_combined_exhausted_via_residual_only():
    # PRICED regime only: F under budget; residual pushes C over.
    # 5e9 + 1e11*0.06 = 1.1e10 > 1e10
    assert is_combined_budget_exhausted(5e9, 0.06, 1e10, LAMBDA_FLOPS_PER_SECOND) is True


def test_residual_alone_cannot_exhaust_under_the_gated_default():
    """With lambda=0 no amount of residual time can trip the combined check.

    That is the point of the gated regime: residual time is bounded by
    residual_wall_time_limit_s instead, and C stays a pure FLOP quantity.
    """
    assert is_combined_budget_exhausted(5e9, 1_000.0, 1e10) is False


def test_combined_not_exhausted_when_budget_nonpositive():
    assert is_combined_budget_exhausted(1e20, 1e20, 0) is False


def test_score_multiplier_floored_at_point_one():
    assert score_multiplier(1e6, 1e10, failed=False) == 0.1


def test_score_multiplier_proportional_between_floor_and_one():
    assert score_multiplier(8e9, 1e10, failed=False) == pytest.approx(0.8)


def test_score_multiplier_uncapped_above_one():
    assert score_multiplier(2e10, 1e10, failed=False) == pytest.approx(2.0)


def test_score_multiplier_failed_is_one():
    assert score_multiplier(1e6, 1e10, failed=True) == 1.0


def test_score_multiplier_nonpositive_budget_is_one():
    assert score_multiplier(1e9, 0, failed=False) == 1.0
