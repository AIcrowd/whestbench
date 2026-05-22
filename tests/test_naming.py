"""Tests for deterministic, seed-driven MLP name generation."""

from __future__ import annotations

import re

import pytest

from whestbench.naming import assign_unique_names, generate_mlp_name

_SLUG_RE = re.compile(r"^[a-z]+(-[a-z]+)+$")


def test_generate_mlp_name_returns_lowercase_hyphenated_slug() -> None:
    """Output must match `^[a-z]+(-[a-z]+)+$` — lowercase letters and hyphens only.

    No apostrophes, no spaces, no titles ("Dr."), no suffixes ("Jr."), no digits.
    Estimator log lines and filenames depend on this shape.
    """
    for seed in (0, 1, 42, 9999, -1, 2**31 - 1):
        name = generate_mlp_name(seed)
        assert _SLUG_RE.match(name), f"seed={seed} produced non-slug name: {name!r}"


def test_generate_mlp_name_is_deterministic_across_calls() -> None:
    """Same seed must yield the same name on repeated calls in the same process."""
    for seed in (0, 1, 42, 123_456, -7):
        assert generate_mlp_name(seed) == generate_mlp_name(seed)


def test_generate_mlp_name_varies_across_seeds() -> None:
    """Different seeds should generally produce different names.

    Not strictly guaranteed (birthday collisions exist), but across 50 small
    integer seeds we expect well above 40 unique results.
    """
    names = {generate_mlp_name(s) for s in range(50)}
    assert len(names) >= 45, f"only {len(names)} unique names across 50 seeds"


def test_generate_mlp_name_does_not_pollute_global_random_state() -> None:
    """Name generation must not leak into the module-level `random` RNG.

    A regression here would silently couple name generation to whatever else
    in the process touches `random.random()`. Use Faker.seed_instance, not
    Faker.seed (the latter mutates global state).
    """
    import random as _random

    _random.seed(12345)
    expected_after = _random.random()

    _random.seed(12345)
    _ = generate_mlp_name(42)
    observed_after = _random.random()

    assert expected_after == observed_after, (
        "name generation perturbed global random state — use Faker().seed_instance(...)"
    )


def test_assign_unique_names_returns_one_per_seed() -> None:
    seeds = [1, 2, 3, 4, 5]
    names = assign_unique_names(seeds)
    assert len(names) == len(seeds)


def test_assign_unique_names_passes_through_when_no_collisions() -> None:
    seeds = [1, 2, 3, 4, 5]
    names = assign_unique_names(seeds)
    assert names == [generate_mlp_name(s) for s in seeds]


def test_assign_unique_names_resolves_collisions_with_numeric_suffix() -> None:
    """When two seeds map to the same first/last pair, the second gets `-2`, third `-3`, etc.

    We force a collision by repeating the same seed three times (degenerate but
    deterministic). The first stays bare, the next two get suffixes.
    """
    seeds = [42, 42, 42]
    names = assign_unique_names(seeds)
    base = generate_mlp_name(42)
    assert names[0] == base
    assert names[1] == f"{base}-2"
    assert names[2] == f"{base}-3"
    assert len(set(names)) == len(names)


def test_assign_unique_names_preserves_input_order_under_collisions() -> None:
    """When a non-leading seed collides with a leading one, the *later* gets the suffix."""
    seeds = [1, 42, 2, 42, 3, 42]
    names = assign_unique_names(seeds)
    # 1, 2, 3 each see their own name unmodified
    assert names[0] == generate_mlp_name(1)
    assert names[2] == generate_mlp_name(2)
    assert names[4] == generate_mlp_name(3)
    # Three occurrences of seed=42 are deduplicated in order
    base = generate_mlp_name(42)
    assert names[1] == base
    assert names[3] == f"{base}-2"
    assert names[5] == f"{base}-3"


@pytest.mark.parametrize("seed", [0, 42, 2**31 - 1, -(2**31), 2**63 - 1])
def test_generate_mlp_name_handles_int_edge_cases(seed: int) -> None:
    """Seed values across the int64 range should not raise."""
    name = generate_mlp_name(seed)
    assert _SLUG_RE.match(name)


# ---------------------------------------------------------------------------
# Faker-version lock-down
# ---------------------------------------------------------------------------
# These tests pin the exact seed -> name mapping to a specific faker version.
# If faker's en_US first/last wordlists change between versions, this test
# trips and forces a deliberate decision: either re-pin faker in
# pyproject.toml and re-bake reference datasets, or revert the upgrade.
#
# The premise of seed-deterministic names is that they're a stable property
# of a WhestBench release. Silent name changes break that premise.
# ---------------------------------------------------------------------------

# Pinned at faker==37.12.0 (see pyproject.toml `"faker~=37.0"`).
_LOCKED_NAMES_FAKER_37 = {
    0: "megan-chang",
    1: "dennis-boone",
    42: "danielle-johnson",
    100: "dustin-gibson",
    9999: "mary-moore",
}


@pytest.mark.parametrize("seed,expected", list(_LOCKED_NAMES_FAKER_37.items()))
def test_locked_down_names_for_pinned_faker_version(seed: int, expected: str) -> None:
    """Lock-down: detect when faker's wordlists change under us.

    To re-baseline after a deliberate faker bump:
      uv run python -c "from whestbench.naming import generate_mlp_name; \\
        print({s: generate_mlp_name(s) for s in (0, 1, 42, 100, 9999)})"
    Update _LOCKED_NAMES_FAKER_37 above with the new values and commit the
    pin bump and the baseline in the same change.
    """
    assert generate_mlp_name(seed) == expected, (
        f"seed={seed} produced {generate_mlp_name(seed)!r}; expected {expected!r}. "
        "faker wordlists may have changed — see test docstring for re-baselining."
    )
