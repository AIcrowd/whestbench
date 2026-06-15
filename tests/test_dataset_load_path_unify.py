"""Tests for the unified _hf_load_split preference chain."""

from __future__ import annotations

from pathlib import Path


def _build_two_config_dataset(tmp_path: Path) -> Path:
    """Bake public(config=default) + holdout(config=holdout) and combine."""
    from whestbench.dataset import create_dataset
    from whestbench.dataset_io import combine_split_datasets

    def bake(name: str, split: str, config: str, seed: int) -> Path:
        out = tmp_path / name
        create_dataset(
            n_mlps=2,
            n_samples=100,
            width=8,
            depth=2,
            mlp_seeds=[seed * 1000 + i for i in range(2)],
            output_path=out,
            split=split,
            config=config,
        )
        return out

    pub = bake("pub", "public", "default", 42)
    hold = bake("hold", "holdout", "holdout", 99)
    out = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=out, write_prepared_arrow=False)
    return out


def test_local_parquet_available_true_for_local_dir(tmp_path: Path):
    from whestbench.dataset import _local_parquet_available

    ds = _build_two_config_dataset(tmp_path)
    assert _local_parquet_available(str(ds), "public", is_local=True) is True
    assert _local_parquet_available(str(ds), "holdout", is_local=True) is True


def test_local_parquet_available_false_for_repo_id():
    from whestbench.dataset import _local_parquet_available

    assert _local_parquet_available("aicrowd/some-repo", "public", is_local=False) is False


def test_local_parquet_available_false_when_split_absent(tmp_path: Path):
    from whestbench.dataset import _local_parquet_available

    ds = _build_two_config_dataset(tmp_path)
    assert _local_parquet_available(str(ds), "nonexistent", is_local=True) is False


def test_load_falls_through_to_manifest_when_fast_path_raises(tmp_path: Path, monkeypatch):
    """If the fast glob path raises, the load still succeeds via the HF
    config-manifest path (never hard-fails)."""
    import whestbench.dataset as wd

    ds_dir = _build_two_config_dataset(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated fast-path failure")

    monkeypatch.setattr(wd, "_load_local_parquet_split", _boom)

    ds = wd.load_dataset(ds_dir, split="holdout")
    assert len(ds) == 2
