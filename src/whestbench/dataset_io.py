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

from datasets import Array2D, Array3D, Features, Sequence, Value

SCHEMA_VERSION = "3.0"
SCHEMA_FORMAT = "hf-datasets-parquet"
SEED_PROTOCOL_NAME = "whestbench_seedsequence_hierarchy"
SEED_PROTOCOL_VERSION = "2.0"

DEFAULT_SPLIT = "public"
HOLDOUT_SPLIT = "holdout"

PARQUET_SUBDIR = "data"
METADATA_FILE = "metadata.json"
README_FILE = "README.md"


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
