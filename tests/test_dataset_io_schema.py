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


def test_validate_config_name_accepts_canonical_names():
    from whestbench.dataset_io import _validate_config_name

    for name in ("default", "holdout", "full", "round-1-eval"):
        assert _validate_config_name(name) == name


def test_validate_config_name_rejects_invalid_names():
    import pytest

    from whestbench.dataset_io import _validate_config_name

    for bad in ("", "Full", "my_config", "1-config", "config--x"):
        with pytest.raises(ValueError, match=r"config name"):
            _validate_config_name(bad)


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


# ---------------------------------------------------------------------------
# prepared_splits validation (Arrow fast path metadata block)
# ---------------------------------------------------------------------------


def _md_with_prepared(**overrides):
    md = _multi_split_md()
    md["prepared_splits"] = {
        "public": {"path": "prepared/public", "format": "save_to_disk"},
        "holdout": {"path": "prepared/holdout", "format": "save_to_disk"},
    }
    md.update(overrides)
    return md


def test_validate_metadata_accepts_prepared_splits():
    from whestbench.dataset_io import validate_metadata

    validate_metadata(_md_with_prepared())  # should not raise


def test_validate_metadata_accepts_prepared_splits_subset():
    """Prepared block may cover only a subset of splits."""
    from whestbench.dataset_io import validate_metadata

    md = _md_with_prepared()
    md["prepared_splits"] = {"public": {"path": "prepared/public", "format": "save_to_disk"}}
    validate_metadata(md)


def test_validate_metadata_accepts_prepared_splits_without_format():
    """Format is optional (defaults to save_to_disk at load time)."""
    from whestbench.dataset_io import validate_metadata

    md = _md_with_prepared()
    md["prepared_splits"]["public"] = {"path": "prepared/public"}
    validate_metadata(md)


def test_validate_metadata_rejects_prepared_splits_non_dict():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _md_with_prepared(prepared_splits=["prepared/public"])
    with pytest.raises(InvalidDatasetError, match=r"prepared_splits.+dict"):
        validate_metadata(md)


def test_validate_metadata_rejects_prepared_splits_unknown_split():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _md_with_prepared()
    md["prepared_splits"]["nonexistent"] = {"path": "prepared/nonexistent"}
    with pytest.raises(InvalidDatasetError, match=r"isn't in this dataset"):
        validate_metadata(md)


def test_validate_metadata_rejects_prepared_splits_entry_non_dict():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _md_with_prepared()
    md["prepared_splits"]["public"] = "prepared/public"  # bare string is invalid
    with pytest.raises(InvalidDatasetError, match=r"must be a dict"):
        validate_metadata(md)


def test_validate_metadata_rejects_prepared_splits_path_non_string():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _md_with_prepared()
    md["prepared_splits"]["public"]["path"] = 42
    with pytest.raises(InvalidDatasetError, match=r"path.+string"):
        validate_metadata(md)


def test_validate_metadata_rejects_prepared_splits_format_non_string():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _md_with_prepared()
    md["prepared_splits"]["public"]["format"] = 1
    with pytest.raises(InvalidDatasetError, match=r"format.+string"):
        validate_metadata(md)


# ---------------------------------------------------------------------------
# _validate_mlp_seeds + seed_protocol 3.0 constants
# ---------------------------------------------------------------------------


def test_validate_mlp_seeds_accepts_canonical_list():
    """Happy path: distinct ints across the full int63 range, including the upper boundary."""
    from whestbench.dataset_io import _validate_mlp_seeds

    seeds = [42, 99, 12345, (1 << 63) - 1]  # last is the max valid value
    _validate_mlp_seeds(seeds, n_mlps=4)  # should not raise


def test_validate_mlp_seeds_rejects_wrong_length():
    import pytest

    from whestbench.dataset_io import _validate_mlp_seeds

    with pytest.raises(ValueError, match=r"length 3.+n_mlps=4"):
        _validate_mlp_seeds([1, 2, 3], n_mlps=4)


def test_validate_mlp_seeds_rejects_empty():
    import pytest

    from whestbench.dataset_io import _validate_mlp_seeds

    with pytest.raises(ValueError, match=r"length 0.+n_mlps=4"):
        _validate_mlp_seeds([], n_mlps=4)


def test_validate_mlp_seeds_rejects_negative():
    import pytest

    from whestbench.dataset_io import _validate_mlp_seeds

    with pytest.raises(ValueError, match=r"mlp_seeds\[1\].+-5"):
        _validate_mlp_seeds([1, -5, 3, 4], n_mlps=4)


def test_validate_mlp_seeds_rejects_too_large():
    import pytest

    from whestbench.dataset_io import _validate_mlp_seeds

    too_big = 2**63  # exactly out of range
    with pytest.raises(ValueError, match=r"mlp_seeds\[2\].+out of range"):
        _validate_mlp_seeds([1, 2, too_big, 4], n_mlps=4)


def test_validate_mlp_seeds_rejects_non_int():
    import pytest

    from whestbench.dataset_io import _validate_mlp_seeds

    with pytest.raises(ValueError, match=r"mlp_seeds\[1\].+str"):
        _validate_mlp_seeds([1, "abc", 3, 4], n_mlps=4)  # type: ignore[list-item]


def test_validate_mlp_seeds_rejects_float():
    import pytest

    from whestbench.dataset_io import _validate_mlp_seeds

    with pytest.raises(ValueError, match=r"mlp_seeds\[0\].+float"):
        _validate_mlp_seeds([1.5, 2, 3, 4], n_mlps=4)  # type: ignore[list-item]


def test_validate_mlp_seeds_rejects_bool():
    """Python bools are ints but should NOT be accepted as seeds (likely a mistake)."""
    import pytest

    from whestbench.dataset_io import _validate_mlp_seeds

    with pytest.raises(ValueError, match=r"mlp_seeds\[0\].+bool"):
        _validate_mlp_seeds([True, 2, 3, 4], n_mlps=4)  # type: ignore[list-item]


def test_validate_mlp_seeds_rejects_duplicates():
    import pytest

    from whestbench.dataset_io import _validate_mlp_seeds

    with pytest.raises(ValueError, match=r"duplicate.+indices"):
        _validate_mlp_seeds([1, 2, 1, 4], n_mlps=4)


def test_seed_protocol_v3_constants_exist():
    from whestbench.dataset_io import SEED_PROTOCOL_NAME_V3, SEED_PROTOCOL_VERSION_V3

    assert SEED_PROTOCOL_NAME_V3 == "whestbench_explicit_per_mlp_seeds"
    assert SEED_PROTOCOL_VERSION_V3 == "3.0"


# ---------------------------------------------------------------------------
# validate_metadata — seed_protocol 3.0 shape
# ---------------------------------------------------------------------------


def _v3_single_split_md(**overrides):
    """Build a valid 3.0 single-split metadata dict."""
    md = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {
            "name": "whestbench_explicit_per_mlp_seeds",
            "version": "3.0",
        },
        "n_mlps": 4,
        "n_samples": 100,
        "width": 8,
        "depth": 2,
        "created_at_utc": "2026-05-26T00:00:00+00:00",
        "hardware": {},
    }
    md.update(overrides)
    return md


def _v3_multi_split_md(**overrides):
    """Build a valid 3.0 multi-split metadata dict."""
    md = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "torch",
        "seed_protocol": {
            "name": "whestbench_explicit_per_mlp_seeds",
            "version": "3.0",
        },
        "n_samples": 100,
        "width": 8,
        "depth": 2,
        "created_at_utc": "2026-05-26T00:00:00+00:00",
        "hardware": {},
        "splits": {
            "public": {
                "n_mlps": 2,
                "created_at_utc": "2026-05-26T00:00:00+00:00",
            },
            "holdout": {
                "n_mlps": 2,
                "created_at_utc": "2026-05-26T00:00:00+00:00",
            },
        },
    }
    md.update(overrides)
    return md


def test_validate_metadata_accepts_v3_single_split():
    from whestbench.dataset_io import validate_metadata

    validate_metadata(_v3_single_split_md())  # should not raise


def test_validate_metadata_accepts_v3_single_split_config_coordinate():
    from whestbench.dataset_io import validate_metadata

    validate_metadata(_v3_single_split_md(split="mini", config="default"))


def test_validate_metadata_rejects_v3_single_split_invalid_config():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _v3_single_split_md(split="mini", config="Full")
    with pytest.raises(InvalidDatasetError, match=r"config.+lowercase"):
        validate_metadata(md)


def test_validate_metadata_accepts_v3_multi_split():
    from whestbench.dataset_io import validate_metadata

    validate_metadata(_v3_multi_split_md())  # should not raise


def test_validate_metadata_accepts_v3_multi_split_config_coordinates():
    from whestbench.dataset_io import validate_metadata

    md = _v3_multi_split_md()
    md["splits"]["public"]["config"] = "default"
    md["splits"]["holdout"]["config"] = "holdout"
    validate_metadata(md)


def test_validate_metadata_rejects_v3_multi_split_non_string_config():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _v3_multi_split_md()
    md["splits"]["holdout"]["config"] = 42
    with pytest.raises(InvalidDatasetError, match=r"config.+string"):
        validate_metadata(md)


def test_validate_metadata_rejects_v3_with_top_level_seed():
    """3.0 single-split must not have a top-level `seed` field."""
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _v3_single_split_md(seed=42)
    with pytest.raises(InvalidDatasetError, match=r"seed_protocol.+3\.0.+top-level.+seed"):
        validate_metadata(md)


def test_validate_metadata_rejects_v3_multi_split_with_top_level_seed():
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _v3_multi_split_md(seed=42)
    with pytest.raises(InvalidDatasetError, match=r"seed_protocol.+3\.0.+top-level.+seed"):
        validate_metadata(md)


def test_validate_metadata_rejects_v3_multi_split_with_per_split_seed():
    """3.0 multi-split must not have per-split `seed` fields."""
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _v3_multi_split_md()
    md["splits"]["public"]["seed"] = 42
    with pytest.raises(InvalidDatasetError, match=r"seed_protocol.+3\.0.+splits.+seed"):
        validate_metadata(md)


def test_validate_metadata_rejects_v2_single_split_missing_seed():
    """2.0 single-split MUST have a top-level seed (existing behavior)."""
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "flopscope",
        "seed_protocol": {
            "name": "whestbench_seedsequence_hierarchy",
            "version": "2.0",
        },
        "n_mlps": 4,
        "n_samples": 100,
        # intentionally no `seed` field
        "width": 8,
        "depth": 2,
        "created_at_utc": "2026-05-26T00:00:00+00:00",
        "hardware": {},
    }
    with pytest.raises(InvalidDatasetError, match=r"seed"):
        validate_metadata(md)


def test_validate_metadata_rejects_unknown_seed_protocol_name():
    """Unknown seed_protocol.name is refused with a clear error."""
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _v3_single_split_md()
    md["seed_protocol"]["name"] = "made-up-protocol"
    with pytest.raises(InvalidDatasetError, match=r"seed_protocol.+name"):
        validate_metadata(md)


def test_validate_metadata_v2_single_split_still_validates():
    """Existing 2.0 single-split datasets must continue to validate (backward compat)."""
    from whestbench.dataset_io import validate_metadata

    md = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "flopscope",
        "seed_protocol": {
            "name": "whestbench_seedsequence_hierarchy",
            "version": "2.0",
        },
        "n_mlps": 4,
        "n_samples": 100,
        "seed": 42,
        "width": 8,
        "depth": 2,
        "created_at_utc": "2026-05-26T00:00:00+00:00",
        "hardware": {},
    }
    validate_metadata(md)  # should not raise


def test_validate_metadata_rejects_v2_single_split_missing_n_mlps():
    """2.0 single-split missing n_mlps fails the same check as 3.0."""
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "backend": "flopscope",
        "seed_protocol": {
            "name": "whestbench_seedsequence_hierarchy",
            "version": "2.0",
        },
        # intentionally no `n_mlps` field
        "n_samples": 100,
        "seed": 42,
        "width": 8,
        "depth": 2,
        "created_at_utc": "2026-05-26T00:00:00+00:00",
        "hardware": {},
    }
    with pytest.raises(InvalidDatasetError, match=r"n_mlps"):
        validate_metadata(md)


# ---------------------------------------------------------------------------
# default_split (optional multi-split field)
# ---------------------------------------------------------------------------


def test_validate_metadata_accepts_valid_default_split():
    """A default_split that names one of the splits validates cleanly."""
    from whestbench.dataset_io import validate_metadata

    md = _multi_split_md(default_split="public")
    validate_metadata(md)  # should not raise


def test_validate_metadata_rejects_default_split_not_in_splits():
    """default_split must name one of the dataset's splits."""
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _multi_split_md(default_split="nonexistent")
    with pytest.raises(InvalidDatasetError, match=r"default_split.*not one of"):
        validate_metadata(md)


def test_validate_metadata_rejects_non_string_default_split():
    """default_split must be a string, not a number or list."""
    import pytest

    from whestbench.dataset_io import InvalidDatasetError, validate_metadata

    md = _multi_split_md(default_split=42)
    with pytest.raises(InvalidDatasetError, match=r"default_split.*string"):
        validate_metadata(md)
