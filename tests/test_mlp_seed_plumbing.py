"""mlp.seed: per-MLP deterministic seed derived from spec.seed."""

from __future__ import annotations

from whestbench.domain import MLP
from whestbench.scoring import ContestSpec, make_contest


def test_mlp_has_seed_field():
    """MLP dataclass must carry a seed int."""
    import flopscope.numpy as fnp

    weights = [fnp.array(fnp.zeros((4, 4), dtype=fnp.float32)) for _ in range(2)]
    mlp = MLP(width=4, depth=2, weights=weights, seed=12345)
    assert mlp.seed == 12345


def test_make_contest_assigns_distinct_seeds_per_mlp():
    """Each MLP in the suite must receive a deterministic, distinct seed."""
    spec = ContestSpec(
        width=8,
        depth=2,
        n_mlps=5,
        flop_budget=1_000_000_000,
        ground_truth_samples=100,
        seed=42,
    )
    data = make_contest(spec)
    seeds = [m.seed for m in data.mlps]
    # Per-MLP seeds must be int and distinct.
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == len(seeds), f"seeds not distinct: {seeds}"


def test_make_contest_seeds_reproduce_across_runs():
    """Same spec.seed → same per-MLP seeds across runs."""
    spec = ContestSpec(
        width=8,
        depth=2,
        n_mlps=3,
        flop_budget=1_000_000_000,
        ground_truth_samples=100,
        seed=99,
    )
    data1 = make_contest(spec)
    data2 = make_contest(spec)
    assert [m.seed for m in data1.mlps] == [m.seed for m in data2.mlps]


def test_make_contest_different_spec_seed_yields_different_mlp_seeds():
    """Different spec.seed → different per-MLP seeds (deterministic derivation)."""
    spec_a = ContestSpec(
        width=8, depth=2, n_mlps=3, flop_budget=1_000_000_000, ground_truth_samples=100, seed=1
    )
    spec_b = ContestSpec(
        width=8, depth=2, n_mlps=3, flop_budget=1_000_000_000, ground_truth_samples=100, seed=2
    )
    seeds_a = [m.seed for m in make_contest(spec_a).mlps]
    seeds_b = [m.seed for m in make_contest(spec_b).mlps]
    assert seeds_a != seeds_b


def test_mlp_seed_roundtrips_through_subprocess_payload():
    """MLP.seed survives serialization through the runner wire protocol."""
    import flopscope.numpy as fnp

    from whestbench.runner import _mlp_to_payload  # noqa: PLC0415
    from whestbench.subprocess_worker import _payload_to_mlp  # noqa: PLC0415

    weights = [fnp.array(fnp.zeros((4, 4), dtype=fnp.float32)) for _ in range(2)]
    original = MLP(width=4, depth=2, weights=weights, seed=7777)
    payload = _mlp_to_payload(original)
    restored = _payload_to_mlp(payload)
    assert restored.seed == 7777
