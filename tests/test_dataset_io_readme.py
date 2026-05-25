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


def test_readme_includes_title():
    out = generate_readme(
        _flopscope_metadata(),
        split="public",
        ds_size=4,
        repo_id="aicrowd/arc-whestbench-2026",
        revision="v1",
    )
    assert "WhestBench Test Dataset" in out


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


def test_readme_placeholder_repo_id_default():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert "<your-repo>" in out
    assert 'revision="main"' in out


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


def test_readme_includes_dataset_card_yaml_front_matter():
    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    assert out.startswith("---\n")
    assert "license: cc-by-4.0" in out
    assert "tags:" in out
    assert "whestbench" in out


def test_readme_renders_loadable_as_dataset_card():
    from huggingface_hub import DatasetCard

    out = generate_readme(_flopscope_metadata(), split="public", ds_size=4)
    card = DatasetCard(out)
    assert card.text is not None
