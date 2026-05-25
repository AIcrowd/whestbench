"""Tests for dataset_io schema constants and Features factory."""

from __future__ import annotations


def test_schema_version_is_3_0():
    from whestbench.dataset_io import SCHEMA_VERSION

    assert SCHEMA_VERSION == "3.0"


def test_schema_format_is_hf_datasets_parquet():
    from whestbench.dataset_io import SCHEMA_FORMAT

    assert SCHEMA_FORMAT == "hf-datasets-parquet"


def test_default_split_name_is_public():
    from whestbench.dataset_io import DEFAULT_SPLIT

    assert DEFAULT_SPLIT == "public"


def test_make_features_returns_features_with_expected_columns():
    from whestbench.dataset_io import make_features

    features = make_features(width=8, depth=2)
    expected_columns = {
        "mlp_id",
        "mlp_name",
        "mlp_seed",
        "weights",
        "all_layer_means",
        "final_means",
        "avg_variance",
        "sampling_budget_breakdown",
    }
    assert set(features.keys()) == expected_columns


def test_make_features_weights_shape_matches_width_and_depth():
    from datasets import Array3D

    from whestbench.dataset_io import make_features

    features = make_features(width=4, depth=3)
    weights = features["weights"]
    assert isinstance(weights, Array3D)
    assert weights.shape == (3, 4, 4)
    assert weights.dtype == "float32"


def test_make_features_all_layer_means_shape():
    from datasets import Array2D

    from whestbench.dataset_io import make_features

    features = make_features(width=4, depth=3)
    means = features["all_layer_means"]
    assert isinstance(means, Array2D)
    assert means.shape == (3, 4)
    assert means.dtype == "float32"


def test_make_features_final_means_is_sequence_of_width():
    from datasets import Sequence, Value

    from whestbench.dataset_io import make_features

    features = make_features(width=5, depth=2)
    final = features["final_means"]
    assert isinstance(final, Sequence)
    assert isinstance(final.feature, Value)
    assert final.feature.dtype == "float32"
    assert final.length == 5
