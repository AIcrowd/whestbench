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


# --- Title & framing ---


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
    assert "WhestBench 2026 — White-Box Activation Estimation" in out


def test_readme_includes_problem_statement():
    """The new self-explanatory framing paragraph must be present."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "white-box activation estimation" in out
    assert "ReLU" in out
    assert "FLOP" in out
    assert "Gaussian" in out


def test_readme_includes_split_aware_lead():
    """The first paragraph names the split."""
    out_pub = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    out_hld = generate_readme(_flopscope_metadata(), split="holdout", ds_size=4)
    assert "`public` split" in out_pub
    assert "`holdout` split" in out_hld


# --- Links ---


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


def test_readme_schema_explains_weights_with_he_init():
    """The weights description should name the activation and initialization."""
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "ReLU" in out
    assert "He initialization" in out
    assert "no biases" in out


def test_readme_schema_explains_avg_variance_formula():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "Var[h_{depth}" in out


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


def test_readme_citation_points_at_challenge_page():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "ARC White-Box Estimation Challenge" in out
    # The citation URL is also in the links bar; this asserts the dedicated section exists.
    assert "## Citation" in out


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
