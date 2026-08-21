"""λ (residual price) is configurable. Default 0 = gated; 1e11 = the Phase 1 priced rate."""

import pytest

from whestbench.budget import (
    LAMBDA_FLOPS_PER_SECOND,
    effective_compute,
    is_combined_budget_exhausted,
)
from whestbench.scoring import ContestSpec


def test_legacy_lambda_constant_still_names_the_phase1_rate():
    """Not the default any more — see test_contestspec_lambda_defaults_to_zero."""
    assert LAMBDA_FLOPS_PER_SECOND == 1e11


def test_effective_compute_default_lambda():
    # Default lambda is 0 (gated regime): residual seconds add nothing, so C == F.
    assert effective_compute(0.0, 1.0) == 0.0
    assert effective_compute(2_500.0, 1.0) == 2_500.0


def test_effective_compute_uses_configured_lambda():
    # 100 FLOPs + 1e10 * 2.0s
    assert effective_compute(100.0, 2.0, lambda_flops_per_second=1e10) == 100.0 + 1e10 * 2.0


def test_combined_exhaustion_respects_lambda():
    assert (
        is_combined_budget_exhausted(0, 1.0, 1_000, lambda_flops_per_second=1.0) is False
    )  # 1.0 <= 1000
    assert (
        is_combined_budget_exhausted(0, 1.0, 1_000, lambda_flops_per_second=10_000.0) is True
    )  # 1e4 > 1000


def _spec(**over):
    base = dict(width=4, depth=2, n_mlps=1, flop_budget=100, ground_truth_samples=10)
    base.update(over)
    return ContestSpec(**base)


def test_contestspec_lambda_defaults_to_zero():
    """Gated, not priced — see whestbench.budget for the two regimes."""
    assert _spec().lambda_flops_per_second == 0.0


def test_contestspec_accepts_custom_lambda():
    assert _spec(lambda_flops_per_second=2e11).lambda_flops_per_second == 2e11


def test_contestspec_accepts_zero_lambda():
    """0 is the default and must validate — it selects the gated regime."""
    _spec(lambda_flops_per_second=0.0).validate()


def test_contestspec_rejects_negative_lambda():
    """A negative rate would pay an estimator for burning wall time."""
    with pytest.raises(ValueError, match="must not be negative"):
        _spec(lambda_flops_per_second=-1.0).validate()


def test_default_lambda_is_zero_gated_regime() -> None:
    """The default is GATED: residual seconds are not priced, so C == F.

    Phase 2 caps residual wall time (residual_wall_time_limit_s) and fails the MLP
    on crossing it, rather than converting seconds into FLOPs. A non-zero default
    here would silently re-price every run against a rule the round does not use.
    """
    from whestbench.budget import DEFAULT_LAMBDA_FLOPS_PER_SECOND, effective_compute

    assert DEFAULT_LAMBDA_FLOPS_PER_SECOND == 0.0
    assert effective_compute(1_000_000, 2.5) == 1_000_000


def test_phase1_rate_still_available_and_unchanged() -> None:
    """Phase 1 rounds must stay reproducible from this codebase."""
    from whestbench.budget import (
        LAMBDA_FLOPS_PER_SECOND,
        PHASE1_LAMBDA_FLOPS_PER_SECOND,
        effective_compute,
    )

    assert PHASE1_LAMBDA_FLOPS_PER_SECOND == 1e11
    # the legacy name has always meant the Phase 1 rate; it must not silently
    # change value just because the default moved off it
    assert LAMBDA_FLOPS_PER_SECOND == PHASE1_LAMBDA_FLOPS_PER_SECOND
    assert effective_compute(1_000_000, 2.5, PHASE1_LAMBDA_FLOPS_PER_SECOND) == 1_000_000 + 2.5e11
