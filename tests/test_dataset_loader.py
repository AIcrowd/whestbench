"""Tests for whestbench.load_dataset, iter_mlps, mlp_at, metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _bake_small(tmp_path: Path, *, split: str = "public") -> Path:
    from whestbench.dataset import create_dataset

    out = tmp_path / f"bake-{split}"
    create_dataset(
        n_mlps=3,
        n_samples=100,
        width=4,
        depth=2,
        mlp_seeds=[42000, 42001, 42002],
        output_path=out,
        split=split,
    )
    return out


def test_load_dataset_returns_datasets_dataset(tmp_path: Path):
    from datasets import Dataset

    from whestbench.dataset import load_dataset  # noqa: E402

    out = _bake_small(tmp_path)
    ds = load_dataset(out)
    assert isinstance(ds, Dataset)
    assert len(ds) == 3


def test_load_dataset_rejects_partial(tmp_path: Path):
    from whestbench.dataset import create_dataset, load_dataset
    from whestbench.dataset_io import InvalidDatasetError

    out = tmp_path / "partial"
    create_dataset(
        n_mlps=10,
        n_samples=50,
        width=4,
        depth=2,
        mlp_seeds=list(range(10)),
        output_path=out,
        mlp_range=(2, 5),
    )
    with pytest.raises(InvalidDatasetError, match="partial"):
        load_dataset(out)


def test_load_dataset_rejects_npz_file(tmp_path: Path):
    """An old schema-2.4 .npz file should produce a helpful error."""
    from whestbench.dataset import load_dataset
    from whestbench.dataset_io import InvalidDatasetError

    npz = tmp_path / "old.npz"
    np.savez(npz, foo=np.array([0]))
    with pytest.raises(InvalidDatasetError):
        load_dataset(npz)


def test_metadata_accessor_returns_parsed_json(tmp_path: Path):
    from whestbench.dataset import load_dataset, metadata

    out = _bake_small(tmp_path)
    ds = load_dataset(out)
    md = metadata(ds)
    assert md["schema_version"] == "3.0"
    # Under seed_protocol 3.0, there is no top-level `seed` field.
    assert "seed" not in md
    assert md["n_mlps"] == 3


def test_metadata_raises_for_bare_hf_load(tmp_path: Path):
    """datasets.load_dataset(...) directly should not get metadata."""
    from datasets import load_dataset as hf_load

    from whestbench.dataset import metadata

    out = _bake_small(tmp_path)
    ds = hf_load(str(out), split="public")
    from whestbench.dataset_io import InvalidDatasetError

    with pytest.raises((KeyError, InvalidDatasetError)):
        metadata(ds)


def test_iter_mlps_yields_validated_mlps(tmp_path: Path):
    from whestbench.dataset import iter_mlps, load_dataset
    from whestbench.domain import MLP

    out = _bake_small(tmp_path)
    ds = load_dataset(out, split="public")
    mlps = list(iter_mlps(ds))
    assert len(mlps) == 3
    for m in mlps:
        assert isinstance(m, MLP)
        assert m.width == 4
        assert m.depth == 2


def test_mlp_at_returns_indexed_mlp(tmp_path: Path):
    from whestbench.dataset import load_dataset, mlp_at

    out = _bake_small(tmp_path)
    ds = load_dataset(out, split="public")
    m0 = mlp_at(ds, 0)
    m2 = mlp_at(ds, 2)
    assert m0.seed != m2.seed
    assert m0.name == ds[0]["mlp_name"]


# -----------------------------------------------------------------------------
# Streaming-mode tests
# -----------------------------------------------------------------------------


def test_load_dataset_streaming_rejects_local_path(tmp_path: Path):
    """streaming=True on a local path is rejected with a clear ValueError."""
    from whestbench.dataset import load_dataset

    out = _bake_small(tmp_path)
    with pytest.raises(ValueError, match="streaming=True is only supported"):
        load_dataset(out, streaming=True)


def test_iter_mlps_works_on_iterable_dataset(tmp_path: Path):
    """iter_mlps() accepts IterableDataset (streaming) and yields correct MLPs.

    We construct an IterableDataset from a materialised one and attach
    metadata via the same side-channel load_dataset would use. This proves
    iter_mlps' code path is streaming-compatible without needing network.
    """
    from datasets import IterableDataset

    from whestbench.dataset import _METADATA_BY_DS, iter_mlps, load_dataset

    out = _bake_small(tmp_path)
    materialised = load_dataset(out, split="public")
    md = _METADATA_BY_DS[materialised]

    # Wrap the same rows as a streaming IterableDataset and attach the
    # same metadata. iter_mlps should produce identical MLP objects.
    rows = list(materialised)
    stream = IterableDataset.from_generator(lambda: iter(rows))
    _METADATA_BY_DS[stream] = md

    reference = list(iter_mlps(materialised))
    streamed = list(iter_mlps(stream))
    assert len(reference) == len(streamed) == 3
    for ref_mlp, str_mlp in zip(reference, streamed):
        assert ref_mlp.name == str_mlp.name
        assert ref_mlp.seed == str_mlp.seed
        assert ref_mlp.width == str_mlp.width
        assert ref_mlp.depth == str_mlp.depth
        ref_w = np.stack([np.asarray(w) for w in ref_mlp.weights])
        str_w = np.stack([np.asarray(w) for w in str_mlp.weights])
        np.testing.assert_array_equal(ref_w, str_w)


def test_mlp_at_on_iterable_dataset_raises_typeerror():
    """mlp_at() raises TypeError on IterableDataset (no random-access contract)."""
    from datasets import IterableDataset

    from whestbench.dataset import mlp_at

    def _gen():
        yield {"mlp_id": 0, "mlp_name": "test", "mlp_seed": 1, "weights": [[[0.0]]]}

    stream = IterableDataset.from_generator(_gen)
    with pytest.raises(TypeError, match="streaming IterableDataset"):
        mlp_at(stream, 0)


def test_iter_mlps_rejects_iterable_dataset_dict():
    """iter_mlps() rejects IterableDatasetDict with a clear error."""
    from datasets import IterableDataset, IterableDatasetDict

    from whestbench.dataset import iter_mlps

    def _gen():
        yield {"mlp_id": 0, "mlp_name": "test", "mlp_seed": 1, "weights": [[[0.0]]]}

    idd = IterableDatasetDict({"public": IterableDataset.from_generator(_gen)})
    with pytest.raises(TypeError, match="multi-split"):
        list(iter_mlps(idd))  # type: ignore[arg-type]


def test_metadata_works_on_iterable_dataset(tmp_path: Path):
    """whestbench.metadata() retrieves attached metadata for IterableDataset too."""
    from datasets import IterableDataset

    from whestbench.dataset import _METADATA_BY_DS, load_dataset, metadata

    out = _bake_small(tmp_path)
    materialised = load_dataset(out, split="public")
    md = _METADATA_BY_DS[materialised]

    rows = list(materialised)
    stream = IterableDataset.from_generator(lambda: iter(rows))
    _METADATA_BY_DS[stream] = md

    md_via_api = metadata(stream)
    assert md_via_api["n_mlps"] == 3
    assert md_via_api["width"] == 4
    assert md_via_api["depth"] == 2
