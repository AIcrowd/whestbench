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
    split: str,
    ds_size: int,
    repo_id: "str | None" = None,
    revision: "str | None" = None,
) -> str:
    """Render the HF dataset card README.

    Reads templates/dataset_card.md.j2, renders with the metadata + context,
    wraps with HF-standard YAML front-matter via DatasetCard.
    """
    template_path = files("whestbench") / "templates" / "dataset_card.md.j2"
    body = Template(template_path.read_text()).render(
        metadata=metadata,
        split=split,
        ds_size=ds_size,
        repo_id=repo_id or "<your-repo>",
        revision=revision or "main",
    )

    card_data = DatasetCardData(
        license="cc-by-4.0",
        language=["code"],
        tags=["whestbench", "alignment", "neural-network-statistics"],
        task_categories=["other"],
        pretty_name=metadata.get("pretty_name", "WhestBench Dataset"),
        size_categories=[_size_category(ds_size)],
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
