"""Create, save, and load whestbench evaluation datasets (schema 3.0)."""

from __future__ import annotations

import dataclasses
import json
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, overload

if TYPE_CHECKING:
    from datasets import DatasetDict

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset

from .dataset_io import (
    DEFAULT_SPLIT,
    SCHEMA_FORMAT,
    SCHEMA_VERSION,
    SEED_PROTOCOL_NAME,
    SEED_PROTOCOL_VERSION,
    InvalidDatasetError,
    make_features,
    read_metadata,
    validate_metadata,
    write_dataset_dir,
)
from .domain import MLP
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint
from .naming import assign_unique_names
from .scoring import _normalize_sampling_budget_breakdown
from .simulation import sample_layer_statistics, sample_layer_statistics_chunk_count

# Metadata side-channel: associates a Dataset/DatasetDict with its
# metadata.json contents without mutating the object itself.
#
# Dataset is hashable -> use a WeakKeyDictionary.
# DatasetDict inherits from dict (unhashable) but IS weakref-able -> use an
# id()-keyed plain dict with a weakref finalizer for cleanup.
_METADATA_BY_DS: "weakref.WeakKeyDictionary[Dataset, Dict[str, Any]]" = weakref.WeakKeyDictionary()
_METADATA_BY_DSD: "Dict[int, Dict[str, Any]]" = {}  # keyed by id(DatasetDict)
_DSD_REFS: "Dict[int, weakref.ref]" = {}  # keeps finalizer alive


def _register_dsd_metadata(dsd: "DatasetDict", md: "Dict[str, Any]") -> None:
    """Store metadata for a DatasetDict, cleaning up when the object is GC'd."""
    oid = id(dsd)
    _METADATA_BY_DSD[oid] = md

    def _cleanup(ref: "weakref.ref", _oid: int = oid) -> None:
        _METADATA_BY_DSD.pop(_oid, None)
        _DSD_REFS.pop(_oid, None)

    _DSD_REFS[oid] = weakref.ref(dsd, _cleanup)


def _get_dsd_metadata(dsd: "DatasetDict") -> "Dict[str, Any] | None":
    return _METADATA_BY_DSD.get(id(dsd))


def _resolve_mlp_range(n_mlps: int, mlp_range: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if mlp_range is None:
        return (0, n_mlps)
    start, end = mlp_range
    if not (0 <= start < end <= n_mlps):
        raise ValueError(
            f"mlp_range {mlp_range!r} invalid for n_mlps={n_mlps}; need 0 <= start < end <= n_mlps."
        )
    return (start, end)


def create_dataset(
    *,
    n_mlps: int,
    n_samples: int,
    width: int,
    depth: int,
    seed: Optional[int] = None,
    output_path: "Path | str",
    split: str = DEFAULT_SPLIT,
    mlp_range: Optional[Tuple[int, int]] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Path:
    """Generate MLPs, compute ground-truth, and write a schema-3.0 dataset directory.

    Output is a directory with data/<split>-00000-of-00001.parquet, metadata.json,
    README.md. Raises FileExistsError if output_path already exists.

    If `mlp_range=(start, end)` is set, only MLPs in [start, end) are generated.
    Output metadata is marked is_partial=true. Run merge_datasets to combine.

    Bit-equivalent property: a worker baking slice [a, b) of a logical dataset of
    size N produces the same rows as the corresponding slice of a single-host bake
    of size N.
    """
    output_path = Path(output_path)
    start, end = _resolve_mlp_range(n_mlps, mlp_range)

    seed_sequence = (
        fnp.random.SeedSequence() if seed is None else fnp.random.SeedSequence(int(seed))
    )
    stream_seed = seed_sequence.spawn(3 * n_mlps)

    # Phase 1: generate MLPs in the slice
    mlps: List[MLP] = []
    for slice_idx, i in enumerate(range(start, end)):
        weight_stream = fnp.random.default_rng(stream_seed[3 * i])
        estimator_seed_i = int(stream_seed[3 * i + 2].generate_state(1)[0])
        mlps.append(sample_mlp(width, depth, weight_stream, seed=estimator_seed_i))
        if progress is not None:
            progress({"phase": "generating", "completed": slice_idx + 1, "total": end - start})

    # Names: derived from ALL logical seeds, then sliced. Guarantees slice's
    # names match the corresponding slice of a single-host bake.
    all_logical_seeds = [int(stream_seed[3 * i + 2].generate_state(1)[0]) for i in range(n_mlps)]
    all_names = assign_unique_names(all_logical_seeds)
    slice_names = all_names[start:end]
    mlps = [dataclasses.replace(m, name=n) for m, n in zip(mlps, slice_names)]

    weights_array = np.stack([np.stack(mlp.weights) for mlp in mlps]).astype(np.float32)

    # Phase 2: ground-truth sampling
    all_means_list: List[fnp.ndarray] = []
    final_means_list: List[fnp.ndarray] = []
    avg_variances: List[float] = []
    sampling_budget_breakdowns: List[Dict[str, Any]] = []
    chunks_per_mlp = sample_layer_statistics_chunk_count(width, n_samples)
    total_sampling_chunks = (end - start) * chunks_per_mlp

    for slice_idx, i in enumerate(range(start, end)):
        sample_stream = fnp.random.default_rng(stream_seed[3 * i + 1])
        mlp = mlps[slice_idx]

        def _on_chunk(
            event,
            *,
            mlp_index=slice_idx + 1,
            name=mlp.name,
            chunk_offset=slice_idx * chunks_per_mlp,
        ):
            if progress is None:
                return
            local_completed = int(event.get("completed", 0))
            progress(
                {
                    "phase": "sampling",
                    "completed": chunk_offset + local_completed,
                    "total": total_sampling_chunks,
                    "mlp_index": mlp_index,
                    "mlp_name": name,
                    "n_mlps": end - start,
                    "unit": "chunks",
                }
            )

        with flops.BudgetContext(flop_budget=int(1e15), quiet=True) as sampling_budget:
            with flops.namespace("sampling"):
                with flops.namespace("sample_layer_statistics"):
                    all_means, final_mean, avg_var = sample_layer_statistics(
                        mlp,
                        n_samples,
                        rng=sample_stream,
                        progress=_on_chunk if progress is not None else None,
                    )
        normalized = _normalize_sampling_budget_breakdown(
            sampling_budget.summary_dict(by_namespace=True)
        )
        if normalized is not None:
            sampling_budget_breakdowns.append(normalized)
        all_means_list.append(fnp.asarray(all_means, dtype=fnp.float32))  # pyright: ignore[reportPossiblyUnboundVariable]
        final_means_list.append(fnp.asarray(final_mean, dtype=fnp.float32))  # pyright: ignore[reportPossiblyUnboundVariable]
        avg_variances.append(avg_var)  # pyright: ignore[reportPossiblyUnboundVariable]

    all_layer_means = np.stack(all_means_list).astype(np.float32)
    final_means = np.stack(final_means_list).astype(np.float32)

    ds = Dataset.from_dict(
        {
            "mlp_id": list(range(start, end)),
            "mlp_name": [m.name for m in mlps],
            "mlp_seed": [int(m.seed) for m in mlps],
            "weights": weights_array,
            "all_layer_means": all_layer_means,
            "final_means": final_means,
            "avg_variance": avg_variances,
            "sampling_budget_breakdown": [json.dumps(b) for b in sampling_budget_breakdowns],
        },
        features=make_features(width=width, depth=depth),
    )

    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "format": SCHEMA_FORMAT,
        "backend": "flopscope",
        "seed_protocol": {
            "name": SEED_PROTOCOL_NAME,
            "version": SEED_PROTOCOL_VERSION,
            "seeded": seed is not None,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed_sequence.entropy),
        "n_mlps": end - start,
        "n_samples": n_samples,
        "width": width,
        "depth": depth,
        "hardware": collect_hardware_fingerprint(),
    }

    is_partial = (start, end) != (0, n_mlps)
    if is_partial:
        metadata["is_partial"] = True
        metadata["mlp_range"] = [start, end]
        metadata["total_n_mlps"] = n_mlps

    write_dataset_dir(ds, output_dir=output_path, split=split, metadata=metadata)
    return output_path


@overload
def load_dataset(
    path_or_repo: "Path | str",
    *,
    revision: Optional[str] = ...,
    split: str,
    token: Optional[str] = ...,
) -> "Dataset": ...


@overload
def load_dataset(
    path_or_repo: "Path | str",
    *,
    revision: Optional[str] = ...,
    split: None = ...,
    token: Optional[str] = ...,
) -> "Dataset | DatasetDict": ...


def load_dataset(
    path_or_repo: "Path | str",
    *,
    revision: Optional[str] = None,
    split: Optional[str] = None,
    token: Optional[str] = None,
) -> "Dataset | DatasetDict":
    """Load a whestbench dataset from a local directory or HF Hub repo.

    Returns `Dataset` for single-split datasets, and when `split=` is provided
    for either single- or multi-split. Returns `DatasetDict` for multi-split
    datasets when `split=` is not provided.

    Metadata is attached via a weakref side-channel (accessible through
    `whestbench.metadata()`):
    - Dataset (single-split or split-selected): the single-split-shaped metadata.
    - DatasetDict (multi-split, no split=): the full multi-split metadata with
      the `splits:` dict.
    - Each member Dataset inside a DatasetDict: the merged single-split-shaped
      metadata for that split.

    Args:
        path_or_repo: Local directory path, or HF Hub repo id (e.g.
            "aicrowd/arc-whestbench-2026").
        revision: HF Hub git tag or commit SHA. Ignored for local paths.
        split: Optional split name. Required for multi-split datasets unless
            you want a DatasetDict; defaults to "public" for single-split
            datasets (preserves prior behavior).
        token: HF Hub auth token. Falls back to HF auth cache.

    Raises:
        InvalidDatasetError: missing/malformed/partial metadata, or unknown
            split name for a multi-split dataset.
    """
    import datasets as hf_datasets
    from datasets import DatasetDict

    path_or_repo_str = str(path_or_repo)
    local_candidate = Path(path_or_repo_str)
    is_local = local_candidate.exists()

    # Materialise metadata.json regardless of source.
    if is_local:
        if local_candidate.is_file():
            raise InvalidDatasetError(
                f"{local_candidate} is a file, not a directory. Schema 3.0 "
                f"datasets are directories; if this is a legacy .npz, re-bake "
                f"with `whest dataset bake`."
            )
        md = read_metadata(local_candidate)
    else:
        from huggingface_hub import hf_hub_download

        metadata_path = hf_hub_download(
            repo_id=path_or_repo_str,
            filename="metadata.json",
            repo_type="dataset",
            revision=revision,
            token=token,
        )
        md = json.loads(Path(metadata_path).read_text())

    validate_metadata(md, allow_partial=False)

    loader_path = str(local_candidate) if is_local else path_or_repo_str
    hf_kwargs: Dict[str, Any] = {} if is_local else {"revision": revision, "token": token}

    if "splits" in md:
        # Multi-split path.
        available_splits = sorted(md["splits"].keys())
        if split is not None:
            if split not in available_splits:
                raise InvalidDatasetError(
                    f"split {split!r} not in {{{', '.join(repr(s) for s in available_splits)}}}"
                )
            ds = hf_datasets.load_dataset(loader_path, split=split, **hf_kwargs)
            _METADATA_BY_DS[ds] = _merge_metadata_for_split(md, split)
            return ds
        dsd = DatasetDict()
        for name in available_splits:
            member = hf_datasets.load_dataset(loader_path, split=name, **hf_kwargs)
            _METADATA_BY_DS[member] = _merge_metadata_for_split(md, name)
            dsd[name] = member
        _register_dsd_metadata(dsd, md)
        return dsd

    # Single-split path. Preserve prior behavior: default split = "public".
    effective_split = split if split is not None else DEFAULT_SPLIT
    ds = hf_datasets.load_dataset(loader_path, split=effective_split, **hf_kwargs)
    _METADATA_BY_DS[ds] = md
    return ds


def _merge_metadata_for_split(md: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Project a multi-split metadata dict into a single-split-shaped dict.

    Returns a new dict with top-level common fields plus the requested split's
    per-split fields (n_mlps, seed, etc.) promoted to top level. The `splits:`
    key is dropped.
    """
    merged: Dict[str, Any] = {k: v for k, v in md.items() if k != "splits"}
    merged.update(md["splits"][split])
    return merged


def metadata(
    ds_or_dsd: "Dataset | DatasetDict",
    *,
    split: "str | None" = None,
) -> Dict[str, Any]:
    """Return the metadata.json contents attached to a Dataset or DatasetDict.

    For a DatasetDict (multi-split, no split= at load time): returns the full
    multi-split metadata dict (with `splits:`). Pass `split="X"` to get a
    single-split-shaped merged dict for split X.

    For a Dataset (single-split or `load_dataset(..., split=X)`): returns the
    single-split-shaped metadata. `split=` is rejected with a TypeError — the
    Dataset's split is fixed at load time and cannot be re-selected here.

    Raises:
        InvalidDatasetError: if no metadata is attached, or if `split=` names
            a split that's not in the DatasetDict.
        TypeError: if `split=` is passed for a Dataset.
    """
    from datasets import DatasetDict

    if isinstance(ds_or_dsd, DatasetDict):
        md = _get_dsd_metadata(ds_or_dsd)
        if md is None:
            raise InvalidDatasetError(
                "no metadata is attached to this DatasetDict; "
                "did you load it via whestbench.load_dataset(...)?"
            )
        if split is None:
            return md
        if "splits" not in md or split not in md["splits"]:
            raise InvalidDatasetError(
                f"split {split!r} not in {sorted(md.get('splits', {}).keys())}"
            )
        return _merge_metadata_for_split(md, split)

    # ds_or_dsd is a Dataset → metadata is already single-split-shaped.
    if split is not None:
        raise TypeError(
            "metadata() does not accept `split=` for a Dataset; the split is "
            "implicit in the loaded Dataset. If you want to filter by split "
            "name, pass a DatasetDict (load_dataset without `split=` for "
            "multi-split datasets)."
        )
    md = _METADATA_BY_DS.get(ds_or_dsd)
    if md is None:
        raise InvalidDatasetError(
            "no metadata is attached to this Dataset; "
            "did you load it via whestbench.load_dataset(...)?"
        )
    return md


def iter_mlps(ds: Dataset) -> Iterator[MLP]:
    """Iterate over a Dataset, yielding one MLP per row."""
    from datasets import DatasetDict

    if isinstance(ds, DatasetDict):
        raise TypeError(
            "iter_mlps requires a single Dataset; for multi-split datasets, "
            "call load_dataset(..., split='<name>') first or use ds[split]."
        )
    for row in ds:
        yield MLP.from_row(row)


def mlp_at(ds: Dataset, index: int) -> MLP:
    """Return the MLP at `index` in the Dataset."""
    from datasets import DatasetDict

    if isinstance(ds, DatasetDict):
        raise TypeError(
            "mlp_at requires a single Dataset; for multi-split datasets, "
            "call load_dataset(..., split='<name>') first or use ds[split]."
        )
    return MLP.from_row(ds[index])
