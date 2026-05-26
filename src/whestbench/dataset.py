"""Create, save, and load whestbench evaluation datasets (schema 3.0)."""

from __future__ import annotations

import dataclasses
import json
import secrets
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, overload

if TYPE_CHECKING:
    from datasets import DatasetDict, IterableDataset, IterableDatasetDict

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset, IterableDataset, IterableDatasetDict

from ._provenance import flopscope_version, whestbench_version
from .dataset_io import (
    DEFAULT_SPLIT,
    SCHEMA_FORMAT,
    SCHEMA_VERSION,
    SEED_PROTOCOL_NAME_V3,
    SEED_PROTOCOL_VERSION_V3,
    InvalidDatasetError,
    _validate_mlp_seeds,
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
# Dataset (and IterableDataset) are hashable -> WeakKeyDictionary works for both.
# DatasetDict / IterableDatasetDict inherit from dict (unhashable) but ARE
# weakref-able -> use an id()-keyed plain dict with a weakref finalizer.
_METADATA_BY_DS: "weakref.WeakKeyDictionary[Dataset | IterableDataset, Dict[str, Any]]" = (
    weakref.WeakKeyDictionary()
)
_METADATA_BY_DSD: "Dict[int, Dict[str, Any]]" = {}  # keyed by id(DatasetDict-like)
_DSD_REFS: "Dict[int, weakref.ref]" = {}  # keeps finalizer alive


def _register_dsd_metadata(dsd: "DatasetDict | IterableDatasetDict", md: "Dict[str, Any]") -> None:
    """Store metadata for a DatasetDict-like, cleaning up when the object is GC'd."""
    oid = id(dsd)
    _METADATA_BY_DSD[oid] = md

    def _cleanup(ref: "weakref.ref", _oid: int = oid) -> None:
        _METADATA_BY_DSD.pop(_oid, None)
        _DSD_REFS.pop(_oid, None)

    _DSD_REFS[oid] = weakref.ref(dsd, _cleanup)


def _get_dsd_metadata(dsd: "DatasetDict | IterableDatasetDict") -> "Dict[str, Any] | None":
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
    mlp_seeds: Optional[List[int]] = None,
    output_path: "Path | str",
    split: str = DEFAULT_SPLIT,
    mlp_range: Optional[Tuple[int, int]] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    **deprecated_kwargs: Any,
) -> Path:
    """Generate MLPs, compute ground-truth, and write a schema-3.0 dataset directory.

    Output is a directory with data/<split>-00000-of-00001.parquet, metadata.json,
    README.md. Raises FileExistsError if output_path already exists.

    Each MLP is seeded by an element of ``mlp_seeds``. When ``mlp_seeds`` is
    omitted, one distinct ``secrets.randbits(63)`` value is generated per MLP.

    If `mlp_range=(start, end)` is set, only MLPs in [start, end) are generated.
    Output metadata is marked is_partial=true. Run merge_datasets to combine.

    Bit-equivalent property: a worker baking slice [a, b) of a logical dataset of
    size N with the same ``mlp_seeds`` list produces the same rows as the
    corresponding slice of a single-host bake of size N.

    Args:
        n_mlps: Total number of MLPs in the logical dataset.
        n_samples: Ground-truth samples per MLP.
        width: Neurons per layer.
        depth: Number of weight matrices.
        mlp_seeds: Per-MLP input seeds (list of ``n_mlps`` distinct int63s).
            Auto-generated when omitted.
        output_path: Destination directory (must not exist).
        split: HF split name for the parquet file.
        mlp_range: ``(start, end)`` to bake a slice of [0, n_mlps).
        progress: Optional callback for progress events.

    Raises:
        TypeError: if the legacy ``seed=`` kwarg is passed.
        ValueError: if ``mlp_seeds`` length or values are invalid.
        FileExistsError: if ``output_path`` already exists.
    """
    # Reject the legacy `seed=` kwarg with a migration hint.
    if "seed" in deprecated_kwargs:
        raise TypeError(
            "seed= is no longer supported in create_dataset. "
            "Use mlp_seeds=[...] to provide explicit per-MLP seeds, "
            "or omit mlp_seeds to auto-generate them."
        )
    if deprecated_kwargs:
        unexpected = ", ".join(repr(k) for k in deprecated_kwargs)
        raise TypeError(f"create_dataset() got unexpected keyword argument(s): {unexpected}")

    output_path = Path(output_path)
    start, end = _resolve_mlp_range(n_mlps, mlp_range)

    # Auto-generate or validate mlp_seeds.
    if mlp_seeds is None:
        # Generate distinct int63 seeds. Collisions are astronomically unlikely
        # (~n^2 / 2^64) but we re-roll defensively. The max_attempts cap prevents
        # an unbounded loop in the pathological case of a broken CSPRNG.
        seen: set = set()
        generated: List[int] = []
        max_attempts = n_mlps * 10
        for _ in range(max_attempts):
            if len(generated) >= n_mlps:
                break
            s = secrets.randbits(63)
            if s not in seen:
                seen.add(s)
                generated.append(s)
        if len(generated) < n_mlps:
            raise RuntimeError(
                f"failed to generate {n_mlps} distinct seeds in {max_attempts} attempts; "
                f"check that secrets.randbits is functioning correctly."
            )
        mlp_seeds = generated
    _validate_mlp_seeds(mlp_seeds, n_mlps)

    # Phase 1: generate MLPs in the slice.
    # Per-MLP SeedSequence: spawn(3) gives [weight_ss, sample_ss, estimator_ss].
    mlps: List[MLP] = []
    for slice_idx, i in enumerate(range(start, end)):
        ss = fnp.random.SeedSequence(mlp_seeds[i]).spawn(3)
        weight_stream = fnp.random.default_rng(ss[0])
        estimator_seed_i = int(ss[2].generate_state(1)[0])
        mlps.append(sample_mlp(width, depth, weight_stream, seed=estimator_seed_i))
        if progress is not None:
            progress({"phase": "generating", "completed": slice_idx + 1, "total": end - start})

    # Names: derived from ALL logical estimator seeds, then sliced. Guarantees
    # slice's names match the corresponding slice of a single-host bake.
    all_logical_seeds = [
        int(fnp.random.SeedSequence(mlp_seeds[i]).spawn(3)[2].generate_state(1)[0])
        for i in range(n_mlps)
    ]
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
        ss = fnp.random.SeedSequence(mlp_seeds[i]).spawn(3)
        sample_stream = fnp.random.default_rng(ss[1])
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
            # Under 3.0, parquet mlp_seed stores the INPUT seed (not derived estimator seed).
            "mlp_seed": mlp_seeds[start:end],
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
            "name": SEED_PROTOCOL_NAME_V3,
            "version": SEED_PROTOCOL_VERSION_V3,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_mlps": end - start,
        "n_samples": n_samples,
        "width": width,
        "depth": depth,
        "hardware": collect_hardware_fingerprint(),
        "whestbench_version": whestbench_version(),
        "flopscope_version": flopscope_version(),
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
    streaming: bool = ...,
) -> "Dataset | IterableDataset": ...


@overload
def load_dataset(
    path_or_repo: "Path | str",
    *,
    revision: Optional[str] = ...,
    split: None = ...,
    token: Optional[str] = ...,
    streaming: bool = ...,
) -> "Dataset | DatasetDict | IterableDataset | IterableDatasetDict": ...


def load_dataset(
    path_or_repo: "Path | str",
    *,
    revision: Optional[str] = None,
    split: Optional[str] = None,
    token: Optional[str] = None,
    streaming: bool = False,
) -> "Dataset | DatasetDict | IterableDataset | IterableDatasetDict":
    """Load a whestbench dataset from a local directory or HF Hub repo.

    Returns `Dataset` for single-split datasets, and when `split=` is provided
    for either single- or multi-split. Returns `DatasetDict` for multi-split
    datasets when `split=` is not provided.

    When ``streaming=True`` (HF Hub only), returns the equivalent streaming
    types: ``IterableDataset`` (single-split or split-selected) or
    ``IterableDatasetDict`` (multi-split, no ``split=``). The metadata
    side-channel works identically for both materialised and streaming
    returns. ``iter_mlps()`` accepts both; ``mlp_at()`` requires a
    materialised dataset and raises ``TypeError`` on streaming inputs.

    Note: streaming datasets cannot currently be used with ``whest run
    --dataset`` because the scoring path uses random-access indexing.

    Metadata is attached via a weakref side-channel (accessible through
    `whestbench.metadata()`):
    - Dataset / IterableDataset (single-split or split-selected): the
      single-split-shaped metadata.
    - DatasetDict / IterableDatasetDict (multi-split, no split=): the full
      multi-split metadata with the `splits:` dict.
    - Each member inside a DatasetDict / IterableDatasetDict: the merged
      single-split-shaped metadata for that split.

    Args:
        path_or_repo: Local directory path, or HF Hub repo id (e.g.
            "aicrowd/arc-whestbench-2026").
        revision: HF Hub git tag or commit SHA. Ignored for local paths.
        split: Optional split name. Required for multi-split datasets unless
            you want a DatasetDict; defaults to "public" for single-split
            datasets (preserves prior behavior).
        token: HF Hub auth token. Falls back to HF auth cache.
        streaming: If True, return HF streaming types (``IterableDataset`` /
            ``IterableDatasetDict``) instead of materialised ones. Only
            supported for HF Hub repos — raises ``ValueError`` for local
            paths.

    Raises:
        InvalidDatasetError: missing/malformed/partial metadata, or unknown
            split name for a multi-split dataset.
        ValueError: ``streaming=True`` combined with a local path.
    """
    import datasets as hf_datasets
    from datasets import DatasetDict, IterableDatasetDict

    path_or_repo_str = str(path_or_repo)
    local_candidate = Path(path_or_repo_str)
    is_local = local_candidate.exists()

    if streaming and is_local:
        raise ValueError(
            "streaming=True is only supported for HF Hub repos, not local paths. "
            "For local datasets, materialise via `load_dataset(local_path)` and "
            "iterate normally."
        )

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
    hf_kwargs: Dict[str, Any] = (
        {} if is_local else {"revision": revision, "token": token, "streaming": streaming}
    )

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
        # Multi-split container. HF gives us a DatasetDict normally; under
        # streaming we assemble an IterableDatasetDict ourselves (HF supports
        # this — IterableDatasetDict is a plain dict subclass).
        dsd: "DatasetDict | IterableDatasetDict" = (
            IterableDatasetDict() if streaming else DatasetDict()
        )
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
    ds_or_dsd: "Dataset | DatasetDict | IterableDataset | IterableDatasetDict",
    *,
    split: "str | None" = None,
) -> Dict[str, Any]:
    """Return the metadata.json contents attached to a Dataset or DatasetDict.

    For a DatasetDict / IterableDatasetDict (multi-split, no split= at load
    time): returns the full multi-split metadata dict (with `splits:`). Pass
    `split="X"` to get a single-split-shaped merged dict for split X.

    For a Dataset / IterableDataset (single-split or
    `load_dataset(..., split=X)`): returns the single-split-shaped metadata.
    `split=` is rejected with a TypeError — the split is fixed at load time
    and cannot be re-selected here.

    Raises:
        InvalidDatasetError: if no metadata is attached, or if `split=` names
            a split that's not in the DatasetDict-like.
        TypeError: if `split=` is passed for a single-Dataset-like.
    """
    from datasets import DatasetDict, IterableDatasetDict

    if isinstance(ds_or_dsd, (DatasetDict, IterableDatasetDict)):
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


def iter_mlps(ds: "Dataset | IterableDataset") -> Iterator[MLP]:
    """Iterate the MLPs in a Dataset, constructing MLP objects per row.

    Accepts both materialised ``Dataset`` and streaming ``IterableDataset``
    inputs — iteration is by ``for row in ds:`` either way, so this works
    transparently on streaming datasets returned by
    ``load_dataset(..., streaming=True)``.

    Looks up the dataset's seed_protocol via the metadata side-channel to
    construct each MLP with the correct estimator-seed derivation. Falls
    back to seed_protocol 2.0 semantics if no metadata is attached (e.g.
    a Dataset constructed manually for tests).
    """
    from datasets import DatasetDict, IterableDatasetDict

    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        raise TypeError(
            "iter_mlps requires a single Dataset / IterableDataset; for "
            "multi-split datasets, call load_dataset(..., split='<name>') "
            "first or use ds[split]."
        )
    md = _METADATA_BY_DS.get(ds, {})
    proto_version = md.get("seed_protocol", {}).get("version", "2.0")
    for row in ds:
        yield MLP.from_row(row, seed_protocol_version=proto_version)


def mlp_at(ds: Dataset, index: int) -> MLP:
    """Return the MLP at `index` in the Dataset.

    Requires a materialised ``Dataset``; streaming ``IterableDataset`` has no
    random-access contract, so this raises ``TypeError`` on streaming inputs.
    To get one MLP at index ``i`` from a streaming dataset, use::

        import itertools
        mlp = next(itertools.islice(iter_mlps(streaming_ds), i, i + 1))

    Cost is O(i) rows scanned — sequential scan to position ``i``.
    """
    from datasets import DatasetDict, IterableDatasetDict

    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        raise TypeError(
            "mlp_at requires a single Dataset; for multi-split datasets, "
            "call load_dataset(..., split='<name>') first or use ds[split]."
        )
    if isinstance(ds, IterableDataset):
        raise TypeError(
            "mlp_at requires a materialised Dataset; streaming IterableDataset "
            "has no random-access contract. To get one MLP at index i from a "
            "streaming dataset, use "
            "`next(itertools.islice(iter_mlps(ds), i, i + 1))` "
            "(cost: O(i) rows scanned), or reload without streaming=True."
        )
    md = _METADATA_BY_DS.get(ds, {})
    proto_version = md.get("seed_protocol", {}).get("version", "2.0")
    return MLP.from_row(ds[index], seed_protocol_version=proto_version)
