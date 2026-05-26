"""On-disk I/O for whestbench datasets in HF-Parquet+sidecar format.

Schema 3.0 layout:
    <dataset_root>/
    ├── data/<split>-NNNNN-of-MMMMM.parquet   # one row per MLP
    ├── metadata.json                         # whestbench provenance
    └── README.md                             # HF dataset card

This module owns the schema constants, the HF Features factory, and
helpers for reading/writing this layout.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import Array2D, Array3D, Dataset, Features, Sequence, Value
from huggingface_hub import DatasetCard, DatasetCardData
from jinja2 import Template

SCHEMA_VERSION = "3.0"
SCHEMA_FORMAT = "hf-datasets-parquet"
SEED_PROTOCOL_NAME = "whestbench_seedsequence_hierarchy"
SEED_PROTOCOL_VERSION = "2.0"
SEED_PROTOCOL_NAME_V3 = "whestbench_explicit_per_mlp_seeds"
SEED_PROTOCOL_VERSION_V3 = "3.0"

DEFAULT_SPLIT = "public"
HOLDOUT_SPLIT = "holdout"

_KNOWN_SEED_PROTOCOLS = {
    (SEED_PROTOCOL_NAME, SEED_PROTOCOL_VERSION),  # 2.0
    (SEED_PROTOCOL_NAME_V3, SEED_PROTOCOL_VERSION_V3),  # 3.0
}

PARQUET_SUBDIR = "data"
METADATA_FILE = "metadata.json"
README_FILE = "README.md"

_SPLIT_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


def _validate_split_name(name: str) -> str:
    """Validate that a split name conforms to the HF Hub convention.

    Allowed: a lowercase ASCII letter, optionally followed by lowercase ASCII
    letters, digits, and single hyphens. Every hyphen must be flanked by
    alphanumeric characters — no leading, trailing, or consecutive hyphens.

    Rejects empty strings, uppercase, underscores, whitespace, dots,
    leading/trailing/consecutive hyphens, and other punctuation.
    """
    if not isinstance(name, str) or name == "":
        raise ValueError("split name must be a non-empty string")
    if not _SPLIT_NAME_RE.match(name):
        raise ValueError(
            f"split name {name!r} does not match [a-z][a-z0-9]*(-[a-z0-9]+)* "
            f"(HF Hub split-name convention: lowercase ASCII, digits, single "
            f"hyphens between alphanumeric runs)."
        )
    return name


def _validate_mlp_seeds(seeds: "list[int]", n_mlps: int) -> None:
    """Validate that mlp_seeds is a list of n_mlps distinct non-negative int63s.

    Each seed must satisfy ``0 <= seed < 2**63`` (positive int64 range, so the
    seeds fit in the parquet ``mlp_seed: int64`` column without wraparound).
    Booleans are rejected even though they are technically int subtypes.
    Duplicate seeds are rejected: two MLPs sharing a seed would be bit-identical,
    almost certainly a user error.

    Raises:
        ValueError: on any rule violation, with a message identifying the
            offending index.
    """
    if not isinstance(seeds, list):
        raise ValueError(f"mlp_seeds must be a list; got {type(seeds).__name__}.")
    if len(seeds) != n_mlps:
        raise ValueError(f"mlp_seeds has length {len(seeds)}, expected n_mlps={n_mlps}.")
    seen: dict[int, int] = {}
    for i, s in enumerate(seeds):
        # bool is a subclass of int; check explicitly.
        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError(f"mlp_seeds[{i}] must be an int; got {type(s).__name__}: {s!r}.")
        if s < 0 or s >= (1 << 63):
            raise ValueError(
                f"mlp_seeds[{i}] = {s} is out of range [0, 2**63). "
                f"Seeds must fit in an int64 column."
            )
        if s in seen:
            raise ValueError(
                f"mlp_seeds contains duplicate at indices [{seen[s]}, {i}]: "
                f"both = {s}. Two MLPs with the same seed are bit-identical; "
                f"likely a user error."
            )
        seen[s] = i


def write_dataset_dir(
    ds: Dataset,
    *,
    output_dir: "Path | str",
    split: str,
    metadata: Dict[str, Any],
    num_shards: int = 1,
) -> Path:
    """Write a Dataset to <output_dir> in the canonical 3-file layout.

    Layout:
        <output_dir>/
        ├── data/<split>-NNNNN-of-MMMMM.parquet
        ├── metadata.json
        └── README.md

    Raises FileExistsError if output_dir already exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir = output_dir / PARQUET_SUBDIR
    data_dir.mkdir()

    if num_shards != 1:
        raise NotImplementedError("multi-shard writes not implemented in schema 3.0")

    parquet_path = data_dir / f"{split}-00000-of-00001.parquet"
    ds.to_parquet(str(parquet_path))

    (output_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    (output_dir / README_FILE).write_text(generate_readme(metadata, split=split, ds_size=len(ds)))
    return output_dir


def _size_category(n: int) -> str:
    """Map an MLP count to HF's standard size_categories label."""
    if n < 1_000:
        return "n<1K"
    if n < 10_000:
        return "1K<n<10K"
    if n < 100_000:
        return "10K<n<100K"
    return "100K<n<1M"


def generate_readme(
    metadata: Dict[str, Any],
    *,
    split: "str | None" = None,
    splits: "dict[str, dict] | None" = None,
    ds_size: int,
    repo_id: "str | None" = None,
    revision: "str | None" = None,
) -> str:
    """Render the HF dataset card README.

    Exactly one of `split` (single-split) or `splits` (multi-split) must be provided.
    """
    if (split is None) == (splits is None):
        raise ValueError("generate_readme requires exactly one of `split` or `splits` to be set.")

    template_path = files("whestbench") / "templates" / "dataset_card.md.j2"
    body = Template(template_path.read_text()).render(
        metadata=metadata,
        split=split,
        splits=splits,
        ds_size=ds_size,
        repo_id=repo_id or "<your-repo>",
        revision=revision or "main",
    )

    tags = [
        "whestbench",
        "alignment",
        "neural-network-statistics",
        "benchmark",
        "white-box",
    ]
    if splits is not None:
        tags.append("multi-split")

    card_data = DatasetCardData(
        license="cc-by-4.0",
        language=["code"],
        tags=tags,
        task_categories=["other"],
        pretty_name=metadata.get(
            "pretty_name", "WhestBench 2026: ARC White-Box Estimation Challenge"
        ),
        size_categories=[_size_category(ds_size)],
        homepage="https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026",
        repository="https://github.com/AIcrowd/whestbench",
    )
    yaml_str = yaml.dump(card_data.to_dict(), default_flow_style=False, allow_unicode=True)
    full_content = f"---\n{yaml_str}---\n\n{body}"
    return str(DatasetCard(full_content))


def make_features(*, width: int, depth: int) -> Features:
    """Return the per-row HF Features schema for a dataset with the given dims.

    Each row is one MLP; shapes are fixed across a single dataset.
    """
    return Features(
        {
            "mlp_id": Value("int32"),
            "mlp_name": Value("string"),
            "mlp_seed": Value("int64"),
            "weights": Array3D(shape=(depth, width, width), dtype="float32"),
            "all_layer_means": Array2D(shape=(depth, width), dtype="float32"),
            "final_means": Sequence(Value("float32"), length=width),
            "avg_variance": Value("float64"),
            "sampling_budget_breakdown": Value("string"),
        }
    )


class InvalidDatasetError(ValueError):
    """Raised when a dataset directory has missing/incompatible metadata."""


def metadata_file_hash(path: "Path | str") -> str:
    """Return the SHA-256 hex digest of metadata.json in the dataset directory.

    Used for logging/reporting to identify a specific bake without loading the
    full dataset. Deterministic given the same dataset directory.
    """
    import hashlib

    metadata_path = Path(path) / METADATA_FILE
    h = hashlib.sha256()
    with metadata_path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def read_metadata(dataset_dir: "Path | str") -> Dict[str, Any]:
    """Read and parse metadata.json from a dataset directory.

    Raises InvalidDatasetError if the file is missing or unparseable.
    Does NOT validate schema_version — call validate_metadata for that.
    """
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / METADATA_FILE
    if not path.is_file():
        raise InvalidDatasetError(
            f"missing metadata.json at {path}. "
            f"This may be a legacy .npz dataset (re-bake with `whest dataset bake`)."
        )
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise InvalidDatasetError(f"metadata.json at {path} is not valid JSON: {exc}") from exc


def validate_metadata(metadata: Dict[str, Any], *, allow_partial: bool = False) -> None:
    """Validate that metadata is a valid whestbench dataset metadata dict.

    Accepts:
    - seed_protocol 2.0 (whestbench_seedsequence_hierarchy): top-level ``seed``
      for single-split, per-split ``seed`` for multi-split.
    - seed_protocol 3.0 (whestbench_explicit_per_mlp_seeds): NO ``seed`` field
      anywhere; the parquet mlp_seed column is canonical.

    The discriminator for single-split vs multi-split is the presence of the
    ``splits`` KEY in metadata, not its value — ``"splits": null`` is invalid
    multi-split, not a fallback to single-split.

    ``allow_partial`` is meaningful only for the single-split shape; multi-split
    datasets cannot be partial.

    Raises:
        InvalidDatasetError: on schema, partial-state, or protocol violations.
    """
    if "schema_version" not in metadata:
        raise InvalidDatasetError(
            "metadata is missing 'schema_version'. "
            "If this is a legacy .npz dataset, re-bake with `whest dataset bake`."
        )
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise InvalidDatasetError(
            f"schema_version is {metadata['schema_version']!r}, this whestbench "
            f"requires {SCHEMA_VERSION!r}. Re-bake with `whest dataset bake`."
        )

    seed_proto = metadata.get("seed_protocol") or {}
    proto_name = seed_proto.get("name")
    proto_version = seed_proto.get("version")
    if (proto_name, proto_version) not in _KNOWN_SEED_PROTOCOLS:
        raise InvalidDatasetError(
            f"unknown seed_protocol name={proto_name!r} / version={proto_version!r}. "
            f"Known protocols: "
            + ", ".join(f"{n!r}@{v!r}" for n, v in sorted(_KNOWN_SEED_PROTOCOLS))
        )

    is_v3 = proto_name == SEED_PROTOCOL_NAME_V3

    if "splits" in metadata:
        # Multi-split shape. Discriminator is key presence; value must still be
        # a non-empty dict (catches null, [], {}).
        splits = metadata["splits"]
        if not isinstance(splits, dict) or not splits:
            raise InvalidDatasetError(
                "multi-split metadata requires 'splits' to be a non-empty dict "
                "with at least one entry."
            )
        if metadata.get("is_partial"):
            raise InvalidDatasetError(
                "multi-split datasets cannot also be partial; got is_partial=true "
                "alongside a 'splits:' dict. Partials are always single-split — "
                "merge each split's partials first, then combine with "
                "`whest dataset combine-splits`."
            )
        if "n_mlps" in metadata:
            raise InvalidDatasetError(
                "multi-split metadata must not have top-level 'n_mlps'; "
                "per-split n_mlps live under splits[<name>]."
            )
        if is_v3:
            if "seed" in metadata:
                raise InvalidDatasetError(
                    "seed_protocol 3.0: multi-split metadata must not have a "
                    "top-level 'seed' field; the parquet mlp_seed column is "
                    "the canonical per-MLP seed record."
                )
        else:
            if "seed" in metadata:
                raise InvalidDatasetError(
                    "multi-split metadata must not have top-level 'seed'; "
                    "per-split seed lives under splits[<name>]."
                )
        for split_name, info in splits.items():
            try:
                _validate_split_name(split_name)
            except ValueError as exc:
                raise InvalidDatasetError(f"invalid split name in splits dict: {exc}") from exc
            if not isinstance(info, dict):
                raise InvalidDatasetError(
                    f"splits[{split_name!r}] must be a dict; got {type(info).__name__}."
                )
            if "n_mlps" not in info:
                raise InvalidDatasetError(
                    f"splits[{split_name!r}] is missing required field 'n_mlps'."
                )
            if is_v3:
                if "seed" in info:
                    raise InvalidDatasetError(
                        f"seed_protocol 3.0: splits[{split_name!r}] must not "
                        f"have a 'seed' field; the parquet mlp_seed column "
                        f"is the canonical per-MLP seed record."
                    )
            else:
                if "seed" not in info:
                    raise InvalidDatasetError(
                        f"splits[{split_name!r}] is missing required field 'seed'."
                    )
        return  # multi-split validated; skip the single-split partial check below

    # Single-split shape.
    if "n_mlps" not in metadata:
        raise InvalidDatasetError("single-split metadata missing required field 'n_mlps'.")
    if is_v3:
        if "seed" in metadata:
            raise InvalidDatasetError(
                "seed_protocol 3.0: single-split metadata must not have a "
                "top-level 'seed' field; the parquet mlp_seed column is "
                "the canonical per-MLP seed record."
            )
    else:
        if "seed" not in metadata:
            raise InvalidDatasetError(
                "seed_protocol 2.0: single-split metadata missing required top-level 'seed' field."
            )

    if metadata.get("is_partial") and not allow_partial:
        mlp_range = metadata.get("mlp_range")
        total = metadata.get("total_n_mlps")
        raise InvalidDatasetError(
            f"this is a partial dataset (mlp_range={mlp_range}, total_n_mlps={total}). "
            f"Run `whest dataset merge <partials> --output <dir>` first."
        )


class MergeIncompatibleError(InvalidDatasetError):
    """Raised when partials have incompatible bake parameters."""


class MergeIncompleteError(InvalidDatasetError):
    """Raised when partials don't cover [0, total_n_mlps) completely."""


class MergeOverlapError(InvalidDatasetError):
    """Raised when partial mlp_range values overlap."""


class MergeCorruptError(InvalidDatasetError):
    """Raised when a partial's row mlp_id values don't match its declared mlp_range."""


_COMMON_FIELDS = ("seed", "n_samples", "width", "depth", "backend")


def merge_datasets(
    input_dirs: "list[Path | str]",
    *,
    output_dir: "Path | str",
) -> Path:
    """Concatenate partial bakes into a single canonical dataset directory.

    Validates that all partials share compatible bake parameters and that
    their mlp_range values cover [0, total_n_mlps) exactly once. Output
    metadata strips per-partial fields and adds hardware_fingerprints and
    merged_at_utc.

    Bit-equivalent to a single-host bake with the same (seed, n_mlps, ...).

    Raises:
        MergeIncompatibleError: partials disagree on schema_version, seed,
            n_samples, width, depth, backend, or total_n_mlps; or any input
            is not a partial.
        MergeIncompleteError: ranges don't cover [0, total_n_mlps) — gaps.
        MergeOverlapError: ranges overlap.
        MergeCorruptError: a partial's row mlp_ids don't match its declared
            mlp_range.
    """
    from datasets import concatenate_datasets
    from datasets import load_dataset as hf_load_dataset

    output_dir = Path(output_dir)
    if not input_dirs:
        raise MergeIncompatibleError("merge_datasets requires at least one input directory")

    partials = []
    for d in input_dirs:
        d = Path(d)
        md = read_metadata(d)
        validate_metadata(md, allow_partial=True)
        if not md.get("is_partial"):
            raise MergeIncompatibleError(
                f"{d} is a complete (non-partial) dataset; merge_datasets only "
                f"accepts partials (is_partial=true)."
            )
        # Determine the split name from the parquet file present in data/
        data_dir = d / PARQUET_SUBDIR
        parquet_files = list(data_dir.glob("*.parquet"))
        if len(parquet_files) != 1:
            raise MergeIncompatibleError(f"{d} has {len(parquet_files)} parquet files; expected 1")
        split_name = parquet_files[0].name.split("-")[0]
        ds = hf_load_dataset(str(d), split=split_name)
        partials.append((d, md, ds, split_name))

    # Validate all share the same common fields
    first_md = partials[0][1]
    for d, md, _, split_name in partials[1:]:
        if split_name != partials[0][3]:
            raise MergeIncompatibleError(
                f"{d} has split {split_name!r}, expected {partials[0][3]!r}"
            )
        if md["schema_version"] != first_md["schema_version"]:
            raise MergeIncompatibleError(
                f"{d}: schema_version {md['schema_version']!r} != {first_md['schema_version']!r}"
            )
        for f in _COMMON_FIELDS:
            if md.get(f) != first_md.get(f):
                raise MergeIncompatibleError(f"{d}: {f}={md.get(f)!r} != {first_md.get(f)!r}")
        if md["total_n_mlps"] != first_md["total_n_mlps"]:
            raise MergeIncompatibleError(
                f"{d}: total_n_mlps={md['total_n_mlps']} != {first_md['total_n_mlps']}"
            )

    total = first_md["total_n_mlps"]
    partials.sort(key=lambda p: p[1]["mlp_range"][0])

    # Check coverage of [0, total)
    expected_start = 0
    for d, md, ds, _ in partials:
        start, end = md["mlp_range"]
        if start < expected_start:
            raise MergeOverlapError(
                f"{d}: mlp_range starts at {start} but previous range ended at {expected_start}"
            )
        if start > expected_start:
            raise MergeIncompleteError(f"gap: no partial covers MLPs [{expected_start}, {start})")
        # Check internal mlp_id consistency
        actual_ids = ds["mlp_id"]
        expected_ids = list(range(start, end))
        if actual_ids != expected_ids:
            raise MergeCorruptError(
                f"{d}: mlp_id values {actual_ids[:5]}... don't match declared range [{start}, {end})"
            )
        expected_start = end

    if expected_start != total:
        raise MergeIncompleteError(f"gap: no partial covers MLPs [{expected_start}, {total})")

    # Concatenate in slice order
    merged_ds = concatenate_datasets([p[2] for p in partials])

    # Reconcile metadata
    merged_md: Dict[str, Any] = {
        k: v
        for k, v in first_md.items()
        if k not in ("is_partial", "mlp_range", "total_n_mlps", "hardware", "created_at_utc")
    }
    merged_md["n_mlps"] = total
    merged_md["created_at_utc"] = min(p[1]["created_at_utc"] for p in partials)
    merged_md["merged_at_utc"] = datetime.now(timezone.utc).isoformat()
    merged_md["hardware_fingerprints"] = [
        {
            **p[1].get("hardware", {}),
            "mlp_range": p[1]["mlp_range"],
            **{k: v for k, v in p[1].items() if k.startswith("cuda_") or k.startswith("mps_")},
        }
        for p in partials
    ]

    write_dataset_dir(
        merged_ds,
        output_dir=output_dir,
        split=partials[0][3],
        metadata=merged_md,
    )
    return output_dir


def combine_split_datasets(
    input_dirs: "list[Path | str]",
    *,
    output_dir: "Path | str",
) -> Path:
    """Combine N complete single-split datasets into a multi-split dataset directory.

    Each input must be a complete (non-partial) single-split schema-3.0 dataset
    directory. Inputs must agree on schema_version, format, backend, seed_protocol,
    width, depth, and n_samples. Split names (inferred from each input's single
    parquet filename) must be pairwise distinct.

    Output is a multi-split dataset directory with one parquet file per input
    split, a multi-split metadata.json (``splits:`` dict), and a rendered README.

    Args:
        input_dirs: Paths to complete single-split dataset directories.
        output_dir: Destination path; must not exist.

    Returns:
        Path to the output directory.

    Raises:
        MergeIncompatibleError: inputs disagree on invariants, contain a partial,
            have duplicate split names, or input list is empty.
        FileExistsError: output_dir already exists.
    """
    import shutil

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")

    if not input_dirs:
        raise MergeIncompatibleError("combine_split_datasets requires at least one input directory")

    # Read + validate each input; collect (path, metadata, split_name, parquet_path).
    entries: list[tuple[Path, Dict[str, Any], str, Path]] = []
    for d in input_dirs:
        d = Path(d)
        md = read_metadata(d)
        validate_metadata(md, allow_partial=True)
        if md.get("is_partial"):
            raise MergeIncompatibleError(
                f"{d}: is a partial dataset; combine_split_datasets requires "
                f"complete single-split inputs. Run `whest dataset merge` first."
            )
        if "splits" in md:
            raise MergeIncompatibleError(
                f"{d}: is already a multi-split dataset; combine_split_datasets "
                f"currently only accepts single-split inputs."
            )
        data_dir = d / PARQUET_SUBDIR
        parquet_files = list(data_dir.glob("*.parquet"))
        if len(parquet_files) != 1:
            raise MergeIncompatibleError(
                f"{d}: expected exactly one parquet file in data/, found {len(parquet_files)}."
            )
        split_name = parquet_files[0].name.split("-")[0]
        try:
            _validate_split_name(split_name)
        except ValueError as exc:
            raise MergeIncompatibleError(
                f"{d}: split name {split_name!r} from parquet filename is invalid: {exc}"
            ) from exc
        entries.append((d, md, split_name, parquet_files[0]))

    # Validate pairwise-distinct split names.
    split_names = [e[2] for e in entries]
    if len(set(split_names)) != len(split_names):
        seen: set[str] = set()
        for n in split_names:
            if n in seen:
                raise MergeIncompatibleError(
                    f"duplicate split name {n!r} across inputs; each input must "
                    f"contribute a distinct split."
                )
            seen.add(n)

    # Validate invariants match across inputs.
    first_md = entries[0][1]
    invariants = ("schema_version", "format", "backend", "n_samples", "width", "depth")
    for d, md, _, _ in entries[1:]:
        for field in invariants:
            if md.get(field) != first_md.get(field):
                raise MergeIncompatibleError(
                    f"{d}: {field}={md.get(field)!r} != {first_md.get(field)!r} "
                    f"(from {entries[0][0]})."
                )
        if md.get("seed_protocol") != first_md.get("seed_protocol"):
            raise MergeIncompatibleError(
                f"{d}: seed_protocol={md.get('seed_protocol')!r} != "
                f"{first_md.get('seed_protocol')!r}."
            )

    # Write atomically: stage in a temp dir, then rename.
    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        staging.mkdir(parents=True)
        data_dst = staging / PARQUET_SUBDIR
        data_dst.mkdir()
        for _d, _md, _split_name, parquet_path in entries:
            shutil.copyfile(parquet_path, data_dst / parquet_path.name)

        # Build the multi-split metadata.
        from .hardware import collect_hardware_fingerprint

        common = {
            k: first_md[k]
            for k in (
                "schema_version",
                "format",
                "backend",
                "seed_protocol",
                "n_samples",
                "width",
                "depth",
            )
        }
        splits_md: Dict[str, Dict[str, Any]] = {}
        for _, md, split_name, _ in entries:
            entry: Dict[str, Any] = {
                "n_mlps": md["n_mlps"],
                "created_at_utc": md["created_at_utc"],
            }
            # Under seed_protocol 2.0, include the per-split `seed` field.
            # Under seed_protocol 3.0, there is no `seed` field (seeds are in parquet).
            if "seed" in md:
                entry["seed"] = md["seed"]
            # Preserve hardware provenance from the bake. merge_datasets-style
            # ``hardware_fingerprints`` (list) takes precedence; otherwise fold
            # the single-host ``hardware`` dict into a one-element list.
            if "hardware_fingerprints" in md:
                entry["hardware_fingerprints"] = md["hardware_fingerprints"]
            elif "hardware" in md:
                entry["hardware_fingerprints"] = [md["hardware"]]
            splits_md[split_name] = entry

        combined_md: Dict[str, Any] = {
            **common,
            "created_at_utc": min(e[1]["created_at_utc"] for e in entries),
            "hardware": collect_hardware_fingerprint(),
            "splits": splits_md,
        }

        # Validate before writing.
        validate_metadata(combined_md)

        (staging / METADATA_FILE).write_text(json.dumps(combined_md, indent=2))

        # Render README. Placeholder repo_id/revision get rewritten at push time.
        ds_size = sum(splits_md[n]["n_mlps"] for n in splits_md)
        (staging / README_FILE).write_text(
            generate_readme(combined_md, splits=splits_md, ds_size=ds_size)
        )

        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return output_dir
