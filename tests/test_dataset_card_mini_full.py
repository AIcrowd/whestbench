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


def test_mini_full_configs_block_emits_mini_first():
    """The YAML frontmatter must contain a configs: block declaring mini before
    full, so the HF Dataset Viewer defaults to mini (the dev split) rather than
    full (alphabetically-first)."""
    from whestbench.dataset_io import generate_readme

    md = _base_metadata()
    rendered = generate_readme(
        md,
        splits=_mini_full_splits(),
        ds_size=1100,
        repo_id="aicrowd/arc-whestbench-public-2026",
        revision="v1-warmup",
    )
    # YAML frontmatter is delimited by leading "---" and the next "---"
    parts = rendered.split("---", 2)
    assert parts[0] == "" and len(parts) >= 3, "missing YAML frontmatter"
    yaml_str = parts[1]
    assert "configs:" in yaml_str, "configs: block missing from YAML frontmatter"
    assert "config_name: default" in yaml_str
    mini_idx = yaml_str.find("split: mini")
    full_idx = yaml_str.find("split: full")
    assert mini_idx >= 0 and full_idx >= 0, "mini/full data_files entries missing"
    assert mini_idx < full_idx, (
        "split order in configs: must be mini before full; got mini at "
        f"{mini_idx}, full at {full_idx}"
    )
    assert "data/mini-*.parquet" in yaml_str
    assert "data/full-*.parquet" in yaml_str


def test_public_holdout_configs_block_emits_public_first():
    """Evaluation datasets (public + holdout) should also get a configs: block
    listing public before holdout (the jobs.yaml insertion order). The exact
    default split for evals isn't critical, but having an explicit order
    pins the Dataset Viewer behavior."""
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
    parts = rendered.split("---", 2)
    yaml_str = parts[1]
    assert "configs:" in yaml_str
    pub_idx = yaml_str.find("split: public")
    hold_idx = yaml_str.find("split: holdout")
    assert pub_idx >= 0 and hold_idx >= 0
    assert pub_idx < hold_idx


def test_single_split_does_not_emit_configs_block():
    """For single-split datasets, HF auto-discovery is fine — no configs:
    block needed. Verify we DON'T emit one in that case."""
    from whestbench.dataset_io import generate_readme

    md = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {"name": "whestbench_explicit_per_mlp_seeds", "version": "3.0"},
        "n_mlps": 100,
        "n_samples": 1_000_000_000,
        "width": 256,
        "depth": 8,
        "created_at_utc": "2026-05-27T00:00:00+00:00",
        "hardware": {},
    }
    rendered = generate_readme(
        md,
        split="public",  # single-split
        splits=None,
        ds_size=100,
        repo_id="some-org/single-split-ds",
        revision="v1",
    )
    parts = rendered.split("---", 2)
    yaml_str = parts[1] if len(parts) >= 3 else ""
    # Either no configs: at all, or no data_files (block could be present but
    # empty — defensive)
    assert "configs:" not in yaml_str, (
        f"single-split README must not emit configs: block; got YAML:\n{yaml_str}"
    )


def test_single_split_non_default_config_emits_configs_block():
    """A single-split bake can declare an explicit non-default HF config."""
    from whestbench.dataset_io import generate_readme

    md = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {"name": "whestbench_explicit_per_mlp_seeds", "version": "3.0"},
        "n_mlps": 100,
        "n_samples": 1_000_000_000,
        "width": 256,
        "depth": 8,
        "created_at_utc": "2026-05-27T00:00:00+00:00",
        "hardware": {},
        "split": "full",
        "config": "full",
    }
    rendered = generate_readme(
        md,
        split="full",
        splits=None,
        ds_size=100,
        repo_id="some-org/full-only-ds",
        revision="v1",
    )
    yaml_str = rendered.split("---", 2)[1]
    assert "configs:" in yaml_str
    assert "config_name: full" in yaml_str
    assert "split: full" in yaml_str
    assert "data/full-*.parquet" in yaml_str


# ---------------------------------------------------------------------------
# Per-split configs (UX1: default_split → independent configs)
# ---------------------------------------------------------------------------


def _base_metadata_with_default_split(default_split: str = "mini"):
    md = _base_metadata()
    md["default_split"] = default_split
    return md


def test_configs_block_with_default_split_emits_one_config_per_split():
    """When metadata declares default_split, each split must live in its own
    config: `default` config contains ONLY the default split, and every other
    split becomes its own named config. This lets HF resolve the requested
    split's data_files in isolation (otherwise load_dataset(repo, split='mini')
    pulls every shard in the repo)."""
    from whestbench.dataset_io import generate_readme

    md = _base_metadata_with_default_split("mini")
    rendered = generate_readme(
        md,
        splits=_mini_full_splits(),
        ds_size=1100,
        repo_id="aicrowd/arc-whestbench-public-2026",
        revision="v1-warmup",
    )
    yaml_str = rendered.split("---", 2)[1]

    # Must have TWO config_name lines: default (= mini) and full.
    assert yaml_str.count("config_name:") == 2, (
        "expected one config per split when default_split is set; got "
        f"{yaml_str.count('config_name:')} configs in YAML:\n{yaml_str}"
    )
    assert "config_name: default" in yaml_str
    assert "config_name: full" in yaml_str

    # The default config must contain ONLY mini.
    default_block_start = yaml_str.find("config_name: default")
    full_block_start = yaml_str.find("config_name: full")
    assert 0 <= default_block_start < full_block_start, (
        f"default config must appear before full config; got default at "
        f"{default_block_start}, full at {full_block_start}"
    )
    default_block = yaml_str[default_block_start:full_block_start]
    assert "split: mini" in default_block
    assert "split: full" not in default_block, (
        "default config must NOT contain split: full; got block:\n" + default_block
    )

    # The full config must contain ONLY full.
    full_block = yaml_str[full_block_start:]
    assert "split: full" in full_block
    assert "split: mini" not in full_block


def test_configs_block_without_default_split_uses_legacy_layout():
    """No default_split → fall back to one `default` config holding every
    split. Preserves backwards-compatibility for datasets baked before the
    field existed."""
    from whestbench.dataset_io import generate_readme

    md = _base_metadata()  # no default_split
    rendered = generate_readme(
        md,
        splits=_mini_full_splits(),
        ds_size=1100,
        repo_id="aicrowd/arc-whestbench-public-2026",
        revision="v1-warmup",
    )
    yaml_str = rendered.split("---", 2)[1]
    assert yaml_str.count("config_name:") == 1, (
        "legacy layout (no default_split) must emit exactly one config; got "
        f"{yaml_str.count('config_name:')} configs in YAML:\n{yaml_str}"
    )
    assert "config_name: default" in yaml_str
    # All splits live under the single `default` config.
    assert "split: mini" in yaml_str
    assert "split: full" in yaml_str


def test_configs_block_with_default_split_holdout_layout():
    """evals (public + holdout) with default_split=public: default config
    holds public, holdout becomes its own named config."""
    from whestbench.dataset_io import generate_readme

    md = _base_metadata()
    md["splits"] = {
        "public": {"n_mlps": 50, "created_at_utc": "2026-05-26T00:00:00+00:00"},
        "holdout": {"n_mlps": 50, "created_at_utc": "2026-05-26T00:00:00+00:00"},
    }
    md["default_split"] = "public"
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
    yaml_str = rendered.split("---", 2)[1]
    assert "config_name: default" in yaml_str
    assert "config_name: holdout" in yaml_str
    default_idx = yaml_str.find("config_name: default")
    holdout_idx = yaml_str.find("config_name: holdout")
    default_block = yaml_str[default_idx:holdout_idx]
    assert "split: public" in default_block
    assert "split: holdout" not in default_block


def test_configs_block_uses_declared_per_split_configs():
    """Config-aware metadata groups data_files by declared config, not split names."""
    from whestbench.dataset_io import generate_readme

    md = _base_metadata()
    md["splits"] = {
        "public": {
            "n_mlps": 50,
            "created_at_utc": "2026-05-26T00:00:00+00:00",
            "config": "default",
        },
        "holdout": {
            "n_mlps": 50,
            "created_at_utc": "2026-05-26T00:00:00+00:00",
            "config": "holdout",
        },
    }
    md["default_split"] = "public"
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
    yaml_str = rendered.split("---", 2)[1]
    assert yaml_str.count("config_name:") == 2
    assert "config_name: default" in yaml_str
    assert "config_name: holdout" in yaml_str
    default_idx = yaml_str.find("config_name: default")
    holdout_idx = yaml_str.find("config_name: holdout")
    default_block = yaml_str[default_idx:holdout_idx]
    holdout_block = yaml_str[holdout_idx:]
    assert "split: public" in default_block
    assert "split: holdout" not in default_block
    assert "split: holdout" in holdout_block
    assert "split: public" not in holdout_block
