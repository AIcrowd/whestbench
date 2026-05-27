"""Regression coverage for the mini+full template branch in dataset_card.md.j2."""

from __future__ import annotations

from types import SimpleNamespace


def _mini_full_splits():
    return {
        "mini": SimpleNamespace(n_mlps=100),
        "full": SimpleNamespace(n_mlps=1000),
    }


def _base_metadata():
    return {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {
            "name": "whestbench_explicit_per_mlp_seeds",
            "version": "3.0",
        },
        "n_samples": 1_000_000_000,
        "width": 256,
        "depth": 8,
        "created_at_utc": "2026-05-27T00:00:00+00:00",
        "hardware": {},
        "splits": {
            "mini": {"n_mlps": 100, "created_at_utc": "2026-05-27T00:00:00+00:00"},
            "full": {"n_mlps": 1000, "created_at_utc": "2026-05-26T00:00:00+00:00"},
        },
    }


def test_mini_full_intro_says_independent_and_disjoint():
    from whestbench.dataset_io import generate_readme

    md = _base_metadata()
    rendered = generate_readme(
        md,
        splits=_mini_full_splits(),
        ds_size=1100,
        repo_id="aicrowd/arc-whestbench-public-2026",
        revision="v1-warmup",
    )
    assert "independent" in rendered.lower()
    assert "disjoint" in rendered.lower()
    assert "100 MLPs" in rendered
    assert "1,000 MLPs" in rendered or "1000 MLPs" in rendered


def test_mini_full_quickstart_uses_split_mini_first():
    from whestbench.dataset_io import generate_readme

    md = _base_metadata()
    rendered = generate_readme(
        md,
        splits=_mini_full_splits(),
        ds_size=1100,
        repo_id="aicrowd/arc-whestbench-public-2026",
        revision="v1-warmup",
    )
    mini_idx = rendered.find('split="mini"')
    full_idx = rendered.find('split="full"')
    assert mini_idx >= 0, 'rendered README missing split="mini" example'
    assert full_idx >= 0, 'rendered README missing split="full" example'
    assert mini_idx < full_idx, 'split="mini" must appear before split="full"'


def test_mini_full_intro_does_not_call_mini_a_subset():
    from whestbench.dataset_io import generate_readme

    md = _base_metadata()
    rendered = generate_readme(
        md,
        splits=_mini_full_splits(),
        ds_size=1100,
        repo_id="aicrowd/arc-whestbench-public-2026",
        revision="v1-warmup",
    )
    # mini must NOT be described as a subset of full. The template must emit the
    # explicit "not a subset" disclaimer phrase. The template uses markdown bold
    # (**not**) so the rendered string is "**not** a subset".
    lower = rendered.lower()
    assert "**not** a subset" in lower, "rendered README must contain the 'not a subset' disclaimer"


def test_holdout_path_unchanged_by_mini_full_addition():
    """Regression: adding the mini+full branch must not alter the public+holdout output."""
    from whestbench.dataset_io import generate_readme

    md = _base_metadata()
    md["splits"] = {
        "public": {"n_mlps": 50, "created_at_utc": "2026-05-26T00:00:00+00:00"},
        "holdout": {"n_mlps": 50, "created_at_utc": "2026-05-26T00:00:00+00:00"},
    }
    rendered = generate_readme(
        md,
        splits={
            "public": SimpleNamespace(n_mlps=50),
            "holdout": SimpleNamespace(n_mlps=50),
        },
        ds_size=100,
        repo_id="aicrowd/arc-whestbench-evals-2026",
        revision="v1-warmup",
    )
    # The existing public+holdout copy must still render — check phrases that
    # only appear in the special-cased public+holdout branch.
    assert "public leaderboard" in rendered.lower(), (
        "public+holdout branch missing 'public leaderboard' language"
    )
    assert "private/final" in rendered.lower(), (
        "public+holdout branch missing 'private/final' language"
    )
