"""Tests for SetupContext.seed and the --seed CLI plumbing."""

from __future__ import annotations

import dataclasses

import pytest

from whestbench.sdk import SetupContext


def test_setup_context_has_seed_field_with_default_zero():
    """SetupContext gains a seed field; default is 0 for backward compat."""
    ctx = SetupContext(
        width=10,
        depth=2,
        flop_budget=100,
        api_version="1.0",
    )
    assert hasattr(ctx, "seed")
    assert ctx.seed == 0


def test_setup_context_accepts_explicit_seed():
    """Seed is settable at construction time."""
    ctx = SetupContext(
        width=10,
        depth=2,
        flop_budget=100,
        api_version="1.0",
        seed=42,
    )
    assert ctx.seed == 42


def test_setup_context_is_frozen_so_seed_cannot_be_mutated():
    """Reproducibility relies on SetupContext being immutable — participants cannot reseed mid-run."""
    ctx = SetupContext(
        width=10,
        depth=2,
        flop_budget=100,
        api_version="1.0",
        seed=42,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.seed = 99  # type: ignore[misc]
