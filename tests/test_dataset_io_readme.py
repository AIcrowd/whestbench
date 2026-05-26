"""Tests for dataset_io.generate_readme."""

from __future__ import annotations

from whestbench.dataset_io import generate_readme


def _flopscope_metadata():
    return {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "flopscope",
        "seed_protocol": {
            "name": "whestbench_seedsequence_hierarchy",
            "version": "2.0",
            "seeded": True,
        },
        "created_at_utc": "2026-05-25T00:00:00+00:00",
        "seed": 42,
        "n_mlps": 4,
        "n_samples": 100_000,
        "width": 8,
        "depth": 2,
        "hardware": {"cpu_brand": "TestCPU"},
        "pretty_name": "WhestBench Test Dataset",
    }


def _torch_metadata():
    md = _flopscope_metadata()
    md.update(
        {
            "backend": "torch",
            "torch_version": "2.5.1",
            "device": "cuda",
            "cuda_device_name": "NVIDIA L40S",
            "mlps_per_batch": 4,
            "chunk_size": 65536,
        }
    )
    return md


def _merged_metadata():
    """A merged-from-parallel-bakes metadata dict."""
    md = _flopscope_metadata()
    # Drop the single-host `hardware` key; merged datasets carry fingerprints instead.
    md.pop("hardware", None)
    md["merged_at_utc"] = "2026-05-25T01:00:00+00:00"
    md["hardware_fingerprints"] = [
        {"cpu_brand": "host-a", "mlp_range": [0, 2]},
        {"cpu_brand": "host-b", "mlp_range": [2, 4]},
    ]
    return md


# --- Logo & framing ---


def test_readme_includes_logo_at_top():
    """The Whest logo should appear above the title, linking to the GitHub repo."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    body = out.split("\n---\n", 1)[1]
    assert (
        '<img src="https://raw.githubusercontent.com/AIcrowd/whestbench/main/assets/logo/logo.png"'
        in body
    )
    # Logo must appear before the title.
    logo_pos = body.find("logo.png")
    title_pos = body.find("# ")
    assert logo_pos != -1 and title_pos != -1
    assert logo_pos < title_pos


def test_readme_includes_organizer_line():
    """An 'Organized by: ARC, AIcrowd' line must appear under the Whest logo."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "Organized by:" in out
    assert "Alignment Research Center (ARC)" in out
    assert "AIcrowd" in out
    # ARC links to alignment.org; AIcrowd links to aicrowd.com
    assert 'href="https://www.alignment.org/"' in out
    assert 'href="https://www.aicrowd.com/"' in out

    body = out.split("\n---\n", 1)[1]
    # Organizer line sits between Whest logo and title
    whest_pos = body.find("logo.png")
    org_pos = body.find("Organized by:")
    title_pos = body.find("# ")
    assert whest_pos < org_pos < title_pos


def test_readme_challenge_badge_uses_aicrowd_brand_color():
    """The Challenge Page badge color is the AIcrowd brand red (#F0524D)."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "img.shields.io/badge/AIcrowd-Challenge_Page-f0524d" in out


def test_readme_includes_title():
    out = generate_readme(
        _flopscope_metadata(),
        split="public",
        ds_size=4,
        repo_id="aicrowd/arc-whestbench-2026",
        revision="v1",
    )
    assert "WhestBench Test Dataset" in out


def test_readme_uses_default_title_when_pretty_name_absent():
    md = _flopscope_metadata()
    del md["pretty_name"]
    out = generate_readme(md, split="public", ds_size=4)
    assert "WhestBench 2026: ARC White-Box Estimation Challenge" in out


def test_readme_includes_problem_statement():
    """The new self-explanatory framing paragraph must be present."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "white-box activation estimation" in out
    assert "ReLU" in out
    assert "FLOP" in out
    assert "Gaussian" in out


def test_readme_split_aware_release_label():
    """Framing reads as a 'Public Dataset Release' vs. 'Holdout Dataset' — not as a 'split'."""
    out_pub = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    out_hld = generate_readme(_flopscope_metadata(), split="holdout", ds_size=4)

    assert "Public Dataset Release for WhestBench 2026" in out_pub
    # Public framing must not call itself the "public split of the evaluation set".
    assert "public` split of the WhestBench 2026 evaluation set" not in out_pub

    assert "Holdout split for WhestBench 2026" in out_hld
    # Holdout points participants at the public release.
    assert "aicrowd/arc-whestbench-2026" in out_hld


# --- Badge row (replaces the bullet-list Links section) ---


def test_readme_links_to_challenge_page():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "aicrowd.com/challenges/arc-white-box-estimation-challenge-2026" in out


def test_readme_links_to_github_repo():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "github.com/AIcrowd/whestbench" in out


def test_readme_links_to_starter_kit():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "github.com/AIcrowd/whest-starterkit" in out


def test_readme_links_to_explorer():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "aicrowd.github.io/whestbench-explorer" in out


def test_readme_links_to_hub_page_with_repo_and_revision():
    out = generate_readme(
        _flopscope_metadata(),
        split="public",
        ds_size=4,
        repo_id="aicrowd/arc-whestbench-2026",
        revision="v1",
    )
    assert "huggingface.co/datasets/aicrowd/arc-whestbench-2026/tree/v1" in out


def test_readme_uses_shields_io_badges_for_all_six_links():
    """Each of the 6 links should be a shields.io badge (`for-the-badge` style)."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    # Each badge URL is a shields.io URL. We check for a unique fragment of each.
    for badge_fragment in (
        "img.shields.io/badge/AIcrowd-Challenge_Page",
        "img.shields.io/badge/GitHub-AIcrowd%2Fwhestbench",
        "img.shields.io/badge/Starter_Kit-whest--starterkit",
        "img.shields.io/badge/MLP_Explorer-Interactive",
        "img.shields.io/badge/FLOP_Tracking-flopscope",
        "View_on_HF_Hub",  # preceded by the URL-encoded 🤗 emoji in the badge URL
    ):
        assert badge_fragment in out, f"badge missing: {badge_fragment!r}"
    # for-the-badge style is consistent across all
    assert out.count("style=for-the-badge") >= 6


def test_readme_flopscope_badge_links_to_github():
    """The flopscope badge wraps a link to the flopscope GitHub repo."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert 'href="https://github.com/AIcrowd/flopscope"' in out


# --- Quick-start snippets ---


def test_readme_includes_load_snippet_with_repo_id():
    out = generate_readme(
        _flopscope_metadata(),
        split="public",
        ds_size=4,
        repo_id="aicrowd/arc-whestbench-2026",
        revision="v1",
    )
    assert "load_dataset(" in out
    assert '"aicrowd/arc-whestbench-2026"' in out
    assert 'revision="v1"' in out
    assert 'split="public"' in out


def test_readme_includes_whestbench_wrapper_snippet():
    out = generate_readme(
        _flopscope_metadata(),
        split="public",
        ds_size=4,
        repo_id="aicrowd/arc-whestbench-2026",
        revision="v1",
    )
    assert "whestbench.load_dataset(" in out
    assert "whestbench.iter_mlps(" in out
    assert "whestbench.metadata(" in out


def test_readme_includes_cli_run_snippet():
    out = generate_readme(
        _flopscope_metadata(),
        split="public",
        ds_size=4,
        repo_id="aicrowd/arc-whestbench-2026",
        revision="v1",
    )
    assert "whest run" in out
    assert "hf://aicrowd/arc-whestbench-2026@v1" in out


def test_readme_invites_starterkit_after_cli_snippet():
    """A call-to-action linking the starter kit appears right after the CLI snippet."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    body = out.split("\n---\n", 1)[1]
    cli_pos = body.find("whest run")
    invite_pos = body.find("New to the challenge?")
    whats_in_pos = body.find("## What's in this dataset")
    # Invitation sits between the CLI snippet and the next section.
    assert cli_pos != -1
    assert invite_pos != -1
    assert whats_in_pos != -1
    assert cli_pos < invite_pos < whats_in_pos
    # The invitation links to the starter kit + mentions flopscope as a learning resource.
    invitation = body[invite_pos:whats_in_pos]
    assert "github.com/AIcrowd/whest-starterkit" in invitation
    assert "flopscope" in invitation


def test_readme_placeholder_repo_id_default():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "<your-repo>" in out
    assert 'revision="main"' in out


# --- Schema descriptions (the de-cryptification) ---


def test_readme_schema_describes_all_eight_columns():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    for col in (
        "mlp_id",
        "mlp_name",
        "mlp_seed",
        "weights",
        "all_layer_means",
        "final_means",
        "avg_variance",
        "sampling_budget_breakdown",
    ):
        assert f"`{col}`" in out, f"column {col!r} missing from schema table"


def test_readme_schema_explains_all_layer_means_as_post_relu_mean():
    """all_layer_means should be described with a concrete expectation formula."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "post-ReLU" in out
    assert "E_{x ~ N(0, I)}" in out
    assert "Monte Carlo" in out


def test_readme_states_the_monte_carlo_sample_count():
    """Production-scale (N = 10⁹) bakes name the MC count in the right places and
    do NOT redundantly add a 'production uses 10⁹' comparison."""
    md = _flopscope_metadata()
    md["n_samples"] = 1_000_000_000  # production-scale N
    out = generate_readme(md, split="public", ds_size=4)
    # Formatted N appears in both schema rows and the dedicated callout.
    assert "1,000,000,000" in out
    # The "## How the ground truth was made" section opens with the N callout.
    section_start = out.find("## How the ground truth was made")
    next_section = out.find("##", section_start + 5)
    section = out[section_start:next_section]
    assert "Monte Carlo with N = 1,000,000,000" in section
    # When the bake IS at production scale, the redundant "production uses N = 10⁹" line
    # must be suppressed — the bake itself is at production scale.
    assert "production WhestBench 2026 release uses" not in out
    assert "production WhestBench 2026 uses" not in out


def test_readme_calls_out_production_target_for_subproduction_bakes():
    """Sub-production bakes (n_samples < 1e9) keep the 'production uses N = 10⁹'
    comparison so participants know what the real release will look like."""
    md = _flopscope_metadata()
    md["n_samples"] = 100_000  # smoke-scale
    out = generate_readme(md, split="public", ds_size=4)
    # The redundant-when-at-production callout SHOULD fire here.
    assert "N = 10⁹" in out
    assert "100,000" in out


def test_readme_summary_table_bolds_n_samples():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "Monte Carlo samples per MLP (N)" in out
    # n_samples in the fixture is 100,000 → table cell has it bolded
    assert "**100,000**" in out


def test_readme_schema_explains_weights_with_he_init():
    """The weights description should name the activation and initialization."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "ReLU" in out
    assert "He initialization" in out
    assert "no biases" in out


def test_readme_schema_explains_avg_variance_formula():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "Var[h_{depth}" in out


def test_readme_avg_variance_does_not_falsely_claim_to_normalize_scoring():
    """avg_variance is computed but NOT consumed by the leaderboard score.

    The previous card text incorrectly said it was used as an MSE normaliser;
    the scoring formula is mse_final · max(0.1, C_m / B_m), no variance term.
    """
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "diagnostic" in out.lower()
    # The card must NOT claim avg_variance normalises the score anywhere.
    assert "normaliser in budget-adjusted scoring" not in out
    assert "normalise MSE across networks" not in out


def test_readme_whats_in_this_dataset_states_monte_carlo_N():
    """The "What's in this dataset" overview must surface the MC sample count N
    used for generating the per-layer means, with the production scale called out."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    section_start = out.find("## What's in this dataset")
    section_end = out.find("## Schema", section_start)
    assert section_start != -1 and section_end != -1
    section = out[section_start:section_end]
    # Item 2 must spell out the metadata N value and reference production N=10⁹.
    assert "N = 100,000" in section  # the fixture's n_samples
    assert "N = 10⁹" in section
    # Variance scalar reframed: no false normaliser claim, diagnostic-only.
    assert "diagnostic provenance" in section


def test_readme_schema_explains_mlp_seed_estimator_use():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "mlp_seed" in out
    assert "predict(mlp: MLP, budget: int)" in out


# --- "Your task" + ground-truth sections ---


def test_readme_explains_estimator_task():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "predict(mlp: MLP, budget: int)" in out
    assert "final_layer_mse" in out or "final-layer MSE" in out
    assert "flopscope" in out


def test_readme_quantifies_per_mlp_budget_and_lambda():
    """B_m = 6.8×10^10 FLOPs and λ = 10^11 FLOPs/s (the deployed value in
    scoring.py:LAMBDA_FLOPS_PER_SECOND, which differs from the paper's
    calibration estimate)."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "6.8 × 10¹⁰" in out
    assert "10¹¹" in out
    # Effective-compute formula C_m = F_m + λ · R_m is spelled out
    assert "F_m + λ" in out
    # The card must NOT use the paper's pre-calibration estimate (7.7 × 10¹¹).
    assert "7.7 × 10¹¹" not in out


def test_readme_failure_path_disables_compute_discount():
    """Failure (over-budget, NaN, wrong shape, guard trip) zeros prediction AND
    forces the multiplier to 1.0 — no compute discount on failures."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "1.0" in out  # forced multiplier
    assert "no compute discount" in out.lower() or "no discount" in out.lower()


def test_readme_no_separate_output_layer_in_weights_description():
    """Paper §1.3: 'every weight matrix is followed by a ReLU; there is no
    additional linear output layer'."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "no separate linear output layer" in out


def test_readme_marks_earlier_layers_as_diagnostic():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    # The card explicitly tags the earlier-layer rows as diagnostic only
    assert "diagnostic" in out.lower()
    assert "all_layer_means[0..depth-2]" in out or "earlier-layer rows" in out


def test_readme_explains_ground_truth_construction():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "float64" in out  # numerical-stability note
    assert "chunked" in out.lower() or "chunk" in out.lower()


# --- Reproducibility section ---


def test_readme_torch_section_only_for_torch_backend():
    cpu = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    gpu = generate_readme(_torch_metadata(), split="public", ds_size=4)
    assert "2.5.1" not in cpu
    assert "2.5.1" in gpu
    assert "L40S" in gpu


def test_readme_includes_rebake_command():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "whest dataset bake" in out
    assert "--seed 42" in out
    assert "--n-mlps 4" in out
    assert "--width 8" in out


# --- Provenance section ---


def test_readme_single_host_provenance_fallback():
    """Non-merged datasets get a single-host fingerprint sentence."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "Single-host bake" in out
    assert "TestCPU" in out


def test_readme_merged_provenance_lists_each_host():
    out = generate_readme(_merged_metadata(), split="public", ds_size=4)
    assert "Assembled from" in out
    assert "host-a" in out
    assert "host-b" in out
    # The MLP-range labels should both appear
    assert "[0, 2)" in out
    assert "[2, 4)" in out


# --- YAML front-matter ---


def test_readme_includes_dataset_card_yaml_front_matter():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert out.startswith("---\n")
    assert "license: cc-by-4.0" in out
    assert "tags:" in out
    assert "whestbench" in out


def test_readme_front_matter_has_homepage_and_repository():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    front_matter = out.split("\n---\n", 1)[0]
    assert (
        "homepage: https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026"
        in front_matter
    )
    assert "repository: https://github.com/AIcrowd/whestbench" in front_matter


def test_readme_front_matter_has_expanded_tags():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    front_matter = out.split("\n---\n", 1)[0]
    for tag in ("whestbench", "alignment", "benchmark", "white-box"):
        assert f"- {tag}" in front_matter


# --- Citation + license ---


def test_readme_citation_is_bibtex_block_attributing_arc_and_aicrowd():
    """Citation section ships a proper BibTeX block citing both organizers."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "## Citation" in out
    # BibTeX fenced block + key entry shape
    assert "```bibtex" in out
    assert "@misc{whestbench2026," in out
    # Title preserved as-is via double-braces
    assert "{{WhestBench 2026: ARC White-Box Estimation Challenge}}" in out
    # Author attribution names BOTH organizations
    assert "{Alignment Research Center}" in out
    assert "{AIcrowd}" in out
    assert "and" in out
    # Standard fields present
    assert "year         = {2026}" in out
    assert "howpublished" in out
    assert "aicrowd.com/challenges/arc-white-box-estimation-challenge-2026" in out


def test_readme_license_section_present():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "## License" in out
    assert "CC-BY-4.0" in out


# --- DatasetCard round-trip ---


def test_readme_renders_loadable_as_dataset_card():
    from huggingface_hub import DatasetCard

    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    card = DatasetCard(out)
    assert card.text is not None


def _multi_split_metadata():
    return {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {"name": "whestbench_seedsequence_hierarchy", "version": "2.0"},
        "n_samples": 1_000_000_000,
        "width": 256,
        "depth": 8,
        "created_at_utc": "2026-05-25T00:00:00+00:00",
        "hardware": {},
        "splits": {
            "public": {"n_mlps": 50, "seed": 1, "created_at_utc": "2026-05-25T00:00:00+00:00"},
            "holdout": {"n_mlps": 50, "seed": 2, "created_at_utc": "2026-05-25T00:00:00+00:00"},
        },
    }


def test_generate_readme_multi_split_renders_eval_dataset_intro():
    from whestbench.dataset_io import generate_readme

    md = _multi_split_metadata()
    out = generate_readme(
        md,
        splits=md["splits"],
        ds_size=100,
        repo_id="aicrowd/arc-whestbench-2026-evals",
        revision="round-1",
    )
    assert "WhestBench 2026 Evaluation Dataset" in out
    assert "public" in out
    assert "holdout" in out
    assert "public leaderboard" in out.lower()
    assert "private/final leaderboard" in out.lower() or "final leaderboard" in out.lower()


def test_generate_readme_multi_split_includes_quick_start_for_both_load_forms():
    from whestbench.dataset_io import generate_readme

    md = _multi_split_metadata()
    out = generate_readme(
        md,
        splits=md["splits"],
        ds_size=100,
        repo_id="aicrowd/arc-whestbench-2026-evals",
        revision="round-1",
    )
    assert 'load_dataset("aicrowd/arc-whestbench-2026-evals", revision="round-1")' in out
    assert 'split="public"' in out


def test_generate_readme_multi_split_has_multi_split_tag_in_yaml_front_matter():
    from whestbench.dataset_io import generate_readme

    md = _multi_split_metadata()
    out = generate_readme(
        md,
        splits=md["splits"],
        ds_size=100,
        repo_id="aicrowd/arc-whestbench-2026-evals",
        revision="round-1",
    )
    front_matter = out.split("---")[1]
    assert "multi-split" in front_matter


def test_generate_readme_multi_split_generic_split_names():
    """If splits aren't {public, holdout}, render generic bullets, no leaderboard prose."""
    from whestbench.dataset_io import generate_readme

    md = _multi_split_metadata()
    md["splits"] = {
        "round-1-a": {"n_mlps": 25, "seed": 1, "created_at_utc": "2026-05-25T00:00:00+00:00"},
        "round-1-b": {"n_mlps": 25, "seed": 2, "created_at_utc": "2026-05-25T00:00:00+00:00"},
    }
    out = generate_readme(
        md,
        splits=md["splits"],
        ds_size=50,
        repo_id="aicrowd/foo",
        revision="main",
    )
    assert "round-1-a" in out
    assert "round-1-b" in out
    assert "leaderboard" not in out.lower()


def test_generate_readme_single_split_public_unchanged_modulo_eval_repo_link():
    """Public-split rendering should still mention 'Public Dataset Release'."""
    from whestbench.dataset_io import generate_readme

    md = _multi_split_metadata()
    del md["splits"]
    md["n_mlps"] = 1000
    md["seed"] = 42
    out = generate_readme(
        md,
        split="public",
        ds_size=1000,
        repo_id="aicrowd/arc-whestbench-2026",
        revision="v1",
    )
    assert "Public Dataset Release" in out
    assert "arc-whestbench-2026-evals" in out


def test_generate_readme_single_split_holdout_unchanged():
    from whestbench.dataset_io import generate_readme

    md = _multi_split_metadata()
    del md["splits"]
    md["n_mlps"] = 50
    md["seed"] = 99
    out = generate_readme(
        md,
        split="holdout",
        ds_size=50,
        repo_id="aicrowd/somewhere",
        revision="v1",
    )
    assert "Holdout split" in out


def test_generate_readme_raises_if_neither_split_nor_splits():
    import pytest

    from whestbench.dataset_io import generate_readme

    md = _multi_split_metadata()
    with pytest.raises(ValueError, match=r"exactly one"):
        generate_readme(md, ds_size=100)


def test_generate_readme_raises_if_both_split_and_splits():
    import pytest

    from whestbench.dataset_io import generate_readme

    md = _multi_split_metadata()
    with pytest.raises(ValueError, match=r"exactly one"):
        generate_readme(
            md,
            split="public",
            splits=md["splits"],
            ds_size=100,
        )


def test_rerender_readme_with_repo_handles_multi_split(tmp_path):
    """The hub.py wrapper detects multi-split dirs and re-renders correctly."""
    from whestbench.dataset import create_dataset
    from whestbench.dataset_io import combine_split_datasets
    from whestbench.hub import _rerender_readme_with_repo

    pub = tmp_path / "pub"
    hold = tmp_path / "hold"
    create_dataset(
        n_mlps=2,
        n_samples=100,
        width=4,
        depth=2,
        mlp_seeds=[1000, 1001],
        output_path=pub,
        split="public",
    )
    create_dataset(
        n_mlps=2,
        n_samples=100,
        width=4,
        depth=2,
        mlp_seeds=[2000, 2001],
        output_path=hold,
        split="holdout",
    )
    combined = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=combined)

    _rerender_readme_with_repo(
        combined,
        repo_id="aicrowd/arc-whestbench-2026-evals",
        revision="round-1",
    )

    readme = (combined / "README.md").read_text()
    assert "Evaluation Dataset" in readme
    assert "aicrowd/arc-whestbench-2026-evals" in readme
    assert "round-1" in readme
    assert "multi-split" in readme  # YAML front-matter tag


def test_rerender_readme_with_repo_handles_single_split(tmp_path):
    """The hub.py wrapper preserves single-split rendering exactly."""
    from whestbench.dataset import create_dataset
    from whestbench.hub import _rerender_readme_with_repo

    out = tmp_path / "single"
    create_dataset(
        n_mlps=2,
        n_samples=100,
        width=4,
        depth=2,
        mlp_seeds=[1000, 1001],
        output_path=out,
    )

    _rerender_readme_with_repo(
        out,
        repo_id="aicrowd/arc-whestbench-2026",
        revision="v1",
    )

    readme = (out / "README.md").read_text()
    assert "Public Dataset Release" in readme
    assert "aicrowd/arc-whestbench-2026" in readme
    assert "v1" in readme


def test_generate_readme_v3_single_split_mentions_protocol():
    """3.0 single-split rendered README mentions the explicit-seed protocol."""
    from whestbench.dataset_io import generate_readme

    md = _v3_single_split_md_for_readme()
    out = generate_readme(
        md,
        split="public",
        ds_size=md["n_mlps"],
        repo_id="aicrowd/example",
        revision="v1",
    )
    assert "whestbench_explicit_per_mlp_seeds" in out or "explicit per-MLP seeds" in out


def test_generate_readme_v3_rebake_command_uses_mlp_seeds():
    """The rebake-command block in the rendered README uses --mlp-seeds, not --seed."""
    from whestbench.dataset_io import generate_readme

    md = _v3_single_split_md_for_readme()
    out = generate_readme(
        md,
        split="public",
        ds_size=md["n_mlps"],
        repo_id="aicrowd/example",
        revision="v1",
    )
    assert "--mlp-seeds" in out
    assert "--seed " not in out and "--seed=" not in out  # legacy flag should not appear


def test_generate_readme_v3_multi_split_mentions_protocol():
    from whestbench.dataset_io import generate_readme

    md = _v3_multi_split_md_for_readme()
    out = generate_readme(
        md,
        splits=md["splits"],
        ds_size=100,
        repo_id="aicrowd/arc-whestbench-2026-evals",
        revision="round-1",
    )
    assert "whestbench_explicit_per_mlp_seeds" in out or "explicit per-MLP seeds" in out


def test_generate_readme_v2_legacy_still_renders():
    """2.0 datasets continue to render (backward compat)."""
    from whestbench.dataset_io import generate_readme

    md = _v2_single_split_md_for_readme()
    out = generate_readme(
        md,
        split="public",
        ds_size=md["n_mlps"],
        repo_id="aicrowd/example",
        revision="v1",
    )
    # 2.0 rebake guidance can still use --seed (or could be tightened later).
    # Just check it renders without crashing.
    assert "Public Dataset Release" in out or "WhestBench" in out


def _v3_single_split_md_for_readme():
    return {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {
            "name": "whestbench_explicit_per_mlp_seeds",
            "version": "3.0",
        },
        "n_mlps": 4,
        "n_samples": 100,
        "width": 4,
        "depth": 2,
        "created_at_utc": "2026-05-26T00:00:00+00:00",
        "hardware": {},
    }


def _v3_multi_split_md_for_readme():
    return {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {
            "name": "whestbench_explicit_per_mlp_seeds",
            "version": "3.0",
        },
        "n_samples": 100,
        "width": 4,
        "depth": 2,
        "created_at_utc": "2026-05-26T00:00:00+00:00",
        "hardware": {},
        "splits": {
            "public": {"n_mlps": 50, "created_at_utc": "2026-05-26T00:00:00+00:00"},
            "holdout": {"n_mlps": 50, "created_at_utc": "2026-05-26T00:00:00+00:00"},
        },
    }


def _v2_single_split_md_for_readme():
    return {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "flopscope",
        "seed_protocol": {
            "name": "whestbench_seedsequence_hierarchy",
            "version": "2.0",
        },
        "n_mlps": 1000,
        "n_samples": 1_000_000_000,
        "seed": 42,
        "width": 256,
        "depth": 8,
        "created_at_utc": "2026-05-25T00:00:00+00:00",
        "hardware": {},
    }
