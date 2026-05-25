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
    # datasets.Sequence's runtime attributes (.feature, .length) aren't in the
    # type stub; pyright flags them despite the isinstance narrow. Ignore.
    assert isinstance(final.feature, Value)  # pyright: ignore[reportAttributeAccessIssue]
    assert final.feature.dtype == "float32"  # pyright: ignore[reportAttributeAccessIssue]
    assert final.length == 5  # pyright: ignore[reportAttributeAccessIssue]


def test_validate_split_name_accepts_canonical_names():
    from whestbench.dataset_io import _validate_split_name

    for name in ("public", "holdout", "my-split", "round-1-eval", "x"):
        assert _validate_split_name(name) == name


def test_validate_split_name_rejects_empty_string():
    import pytest

    from whestbench.dataset_io import _validate_split_name

    with pytest.raises(ValueError, match="empty"):
        _validate_split_name("")


def test_validate_split_name_rejects_uppercase():
    import pytest

    from whestbench.dataset_io import _validate_split_name

    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9\]\*"):
        _validate_split_name("Public")


def test_validate_split_name_rejects_underscore():
    import pytest

    from whestbench.dataset_io import _validate_split_name

    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9\]\*"):
        _validate_split_name("my_split")


def test_validate_split_name_rejects_leading_digit():
    import pytest

    from whestbench.dataset_io import _validate_split_name

    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9\]\*"):
        _validate_split_name("1-split")


def test_validate_split_name_rejects_whitespace_and_special_chars():
    import pytest

    from whestbench.dataset_io import _validate_split_name

    for bad in ("with space", "with.dot", "with/slash", "with\ttab"):
        with pytest.raises(ValueError):
            _validate_split_name(bad)


def test_validate_split_name_rejects_trailing_hyphen():
    import pytest

    from whestbench.dataset_io import _validate_split_name

    for bad in ("a-", "round-1-", "public-"):
        with pytest.raises(ValueError):
            _validate_split_name(bad)


def test_validate_split_name_rejects_consecutive_hyphens():
    import pytest

    from whestbench.dataset_io import _validate_split_name

    for bad in ("a--b", "round--1", "x--y--z"):
        with pytest.raises(ValueError):
            _validate_split_name(bad)


def test_validate_split_name_rejects_digit_only():
    import pytest

    from whestbench.dataset_io import _validate_split_name

    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9\]\*"):
        _validate_split_name("123")


# ---------------------------------------------------------------------------
# validate_metadata — multi-split shape
# ---------------------------------------------------------------------------


def _base_md(**overrides):
    """Build a valid single-split metadata dict for testing."""
    md = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {"name": "whestbench_seedsequence_hierarchy", "version": "2.0"},
        "n_mlps": 4,
        "n_samples": 100,
        "seed": 42,
        "width": 8,
        "depth": 2,
        "created_at_utc": "2026-05-25T00:00:00+00:00",
        "hardware": {},
    }
    md.update(overrides)
    return md


def _multi_split_md(**overrides):
    """Build a valid multi-split metadata dict for testing."""
    md = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {"name": "whestbench_seedsequence_hierarchy", "version": "2.0"},
        "n_samples": 100,
        "width": 8,
        "depth": 2,
        "created_at_utc": "2026-05-25T00:00:00+00:00",
        "hardware": {},
        "splits": {
            "public": {
                "n_mlps": 2,
                "seed": 42,
                "created_at_utc": "2026-05-25T00:00:00+00:00",
            },
            "holdout": {
                "n_mlps": 2,
                "seed": 99,
                "created_at_utc": "2026-05-25T00:00:00+00:00",
            },
        },
    }
    md.update(overrides)
    return md


def test_validate_metadata_accepts_single_split_unchanged():
    from whestbench.dataset_io import validate_metadata

    validate_metadata(_base_md())  # should not raise


def test_validate_metadata_accepts_multi_split():
    from whestbench.dataset_io import validate_metadata

    validate_metadata(_multi_split_md())  # should not raise


def test_validate_metadata_rejects_multi_split_with_top_level_n_mlps():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _multi_split_md(n_mlps=4)  # add forbidden top-level field
    with pytest.raises(InvalidDatasetError, match=r"top-level.+n_mlps"):
        validate_metadata(md)


def test_validate_metadata_rejects_multi_split_with_top_level_seed():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _multi_split_md(seed=42)
    with pytest.raises(InvalidDatasetError, match=r"top-level.+seed"):
        validate_metadata(md)


def test_validate_metadata_rejects_empty_splits_dict():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _multi_split_md(splits={})
    with pytest.raises(InvalidDatasetError, match=r"at least one"):
        validate_metadata(md)


def test_validate_metadata_rejects_split_missing_n_mlps():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _multi_split_md()
    del md["splits"]["public"]["n_mlps"]
    with pytest.raises(InvalidDatasetError, match=r"splits\['public'\].+n_mlps"):
        validate_metadata(md)


def test_validate_metadata_rejects_split_missing_seed():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _multi_split_md()
    del md["splits"]["holdout"]["seed"]
    with pytest.raises(InvalidDatasetError, match=r"splits\['holdout'\].+seed"):
        validate_metadata(md)


def test_validate_metadata_rejects_multi_split_with_is_partial():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _multi_split_md(is_partial=True)
    with pytest.raises(InvalidDatasetError, match=r"partial"):
        validate_metadata(md)


def test_validate_metadata_rejects_invalid_split_name_in_splits_dict():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _multi_split_md()
    md["splits"]["Bad-Name"] = {"n_mlps": 2, "seed": 1}
    with pytest.raises(InvalidDatasetError, match=r"split name.+(lowercase|convention)"):
        validate_metadata(md)


def test_validate_metadata_rejects_splits_null():
    """Discriminator is key presence — null value must NOT fall through to single-split."""
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _multi_split_md(splits=None)
    with pytest.raises(InvalidDatasetError, match=r"non-empty dict"):
        validate_metadata(md)
