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
