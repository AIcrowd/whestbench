"""Deterministic, seed-driven human-readable MLP names.

Each MLP carries a name like ``"john-smith"`` derived deterministically from
its per-MLP seed. Names are reproducible across runs, across CPU/GPU backends,
and across machines — at the ``faker`` version pinned in ``pyproject.toml``.
Bumping ``faker`` is a deliberate operation that requires re-baking reference
datasets; the lock-down test in ``tests/test_naming.py`` trips when faker's
output for ``seed=42`` changes.

Implementation note: a fresh ``Faker()`` instance is built per call and then
seeded via ``seed_instance``. The class-level ``Faker.seed()`` is *not* used —
it mutates the module-level ``random`` state. Per-call instantiation costs
~1 ms/MLP, negligible at the n_mlps scales the evaluator targets.
"""

from __future__ import annotations

import re
from typing import Dict, List

from faker import Faker

# Drop everything that isn't an ASCII lowercase letter, whitespace, or hyphen.
_DROP_NON_SLUG = re.compile(r"[^a-z\s-]")
# Collapse any run of whitespace/hyphens to a single separator.
_COLLAPSE_SEP = re.compile(r"[\s-]+")


def _slugify(text: str) -> str:
    """Lowercase, strip non-letters, collapse separators to single hyphens.

    Used to coerce a faker-produced first/last name fragment into the
    ``[a-z]+(?:-[a-z]+)*`` shape WhestBench uses everywhere a name appears.
    """
    lowered = text.lower()
    no_punct = _DROP_NON_SLUG.sub("", lowered)
    return _COLLAPSE_SEP.sub("-", no_punct).strip("-")


def generate_mlp_name(seed: int) -> str:
    """Return a deterministic, lowercase, hyphenated person-name slug.

    Output matches ``^[a-z]+(-[a-z]+)+$``. Stable across runs and machines at
    the pinned ``faker`` version.

    Args:
        seed: Integer seed (typically ``MLP.seed``).

    Returns:
        Slug like ``"john-smith"``.
    """
    fake = Faker()
    fake.seed_instance(int(seed))
    first = _slugify(fake.first_name())
    last = _slugify(fake.last_name())
    return f"{first}-{last}"


def assign_unique_names(seeds: List[int]) -> List[str]:
    """Build per-MLP names, resolving any within-list collisions in order.

    The first occurrence of a generated name keeps it bare; subsequent
    collisions get ``-2``, ``-3``, ... suffixes appended (e.g.
    ``john-smith``, ``john-smith-2``, ``john-smith-3``).

    Args:
        seeds: Per-MLP seeds, in MLP index order.

    Returns:
        One slug per seed; all entries pairwise distinct.
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for s in seeds:
        base = generate_mlp_name(s)
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}-{count + 1}")
    return out
