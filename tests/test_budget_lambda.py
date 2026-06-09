"""λ (residual penalty rate) is a configurable parameter with a 1e11 default."""

from whestbench.budget import (
    LAMBDA_FLOPS_PER_SECOND,
    effective_compute,
    is_combined_budget_exhausted,
)


def test_default_lambda_is_1e11():
    assert LAMBDA_FLOPS_PER_SECOND == 1e11


def test_effective_compute_default_lambda():
    # 0 FLOPs + 1e11 * 1.0s = 1e11
    assert effective_compute(0.0, 1.0) == 1e11


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
