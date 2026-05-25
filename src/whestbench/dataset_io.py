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
from pathlib import Path
from typing import Any, Dict

from datasets import Array2D, Array3D, Dataset, Features, Sequence, Value

SCHEMA_VERSION = "3.0"
SCHEMA_FORMAT = "hf-datasets-parquet"
SEED_PROTOCOL_NAME = "whestbench_seedsequence_hierarchy"
SEED_PROTOCOL_VERSION = "2.0"

DEFAULT_SPLIT = "public"
HOLDOUT_SPLIT = "holdout"

PARQUET_SUBDIR = "data"
METADATA_FILE = "metadata.json"
README_FILE = "README.md"


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


def generate_readme(
    metadata: Dict[str, Any],
    *,
    split: str,
    ds_size: int,
    repo_id: "str | None" = None,
    revision: "str | None" = None,
) -> str:
    """Placeholder; replaced by Jinja template in Task 5."""
    return f"# {metadata.get('pretty_name', 'WhestBench Dataset')}\n\nSplit: `{split}`, MLPs: {ds_size}\n"


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
    """Validate that metadata is a whestbench schema 3.0 metadata dict.

    Raises InvalidDatasetError with a clear remediation message on any failure.
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
    if seed_proto.get("version") != SEED_PROTOCOL_VERSION:
        raise InvalidDatasetError(
            f"seed_protocol.version is {seed_proto.get('version')!r}, this "
            f"whestbench requires {SEED_PROTOCOL_VERSION!r}. Re-bake."
        )
    if metadata.get("is_partial") and not allow_partial:
        mlp_range = metadata.get("mlp_range")
        total = metadata.get("total_n_mlps")
        raise InvalidDatasetError(
            f"this is a partial dataset (mlp_range={mlp_range}, total_n_mlps={total}). "
            f"Run `whest dataset merge <partials> --output <dir>` first."
        )
