import flopscope.numpy as fnp
import numpy as np
import pytest

from whestbench.domain import MLP


def test_mlp_validate_accepts_valid_mlp() -> None:
    weights = [fnp.zeros((4, 4)) for _ in range(3)]
    mlp = MLP(width=4, depth=3, weights=weights)
    mlp.validate()  # should not raise


def test_mlp_validate_rejects_zero_width() -> None:
    with pytest.raises(ValueError, match="width"):
        MLP(width=0, depth=1, weights=[fnp.zeros((0, 0))]).validate()


def test_mlp_validate_rejects_zero_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        MLP(width=4, depth=0, weights=[]).validate()


def test_mlp_validate_rejects_depth_mismatch() -> None:
    weights = [fnp.zeros((4, 4))]
    mlp = MLP(width=4, depth=2, weights=weights)
    with pytest.raises(ValueError, match="depth"):
        mlp.validate()


def test_mlp_validate_rejects_wrong_weight_shape() -> None:
    weights = [fnp.zeros((4, 3))]
    mlp = MLP(width=4, depth=1, weights=weights)
    with pytest.raises(ValueError, match="shape"):
        mlp.validate()


def test_mlp_name_defaults_to_empty_string() -> None:
    """Construction without an explicit name yields name=''.

    The default keeps every callsite that constructs MLPs directly (tests,
    subprocess worker, scoring module) working unchanged. The evaluator-side
    bake paths (create_dataset, create_dataset_torch, make_contest) populate
    `name` explicitly.
    """
    weights = [fnp.zeros((4, 4)) for _ in range(3)]
    mlp = MLP(width=4, depth=3, weights=weights)
    assert mlp.name == ""


def test_mlp_accepts_explicit_name() -> None:
    """An explicit name kwarg is stored verbatim and is visible to estimators."""
    weights = [fnp.zeros((4, 4)) for _ in range(3)]
    mlp = MLP(width=4, depth=3, weights=weights, name="danielle-johnson")
    assert mlp.name == "danielle-johnson"
    mlp.validate()  # name does not affect validation


def test_mlp_from_row_builds_valid_mlp():
    width, depth = 4, 2
    row = {
        "mlp_seed": 123,
        "mlp_name": "test-name",
        "weights": np.random.default_rng(0)
        .standard_normal((depth, width, width))
        .astype("float32"),
    }
    mlp = MLP.from_row(row)
    assert mlp.width == width
    assert mlp.depth == depth
    assert mlp.seed == 123
    assert mlp.name == "test-name"
    assert len(mlp.weights) == depth
    assert mlp.weights[0].shape == (width, width)


def test_mlp_from_row_accepts_list_of_lists():
    """datasets.Dataset rows may yield nested lists rather than arrays."""
    width, depth = 3, 2
    row = {
        "mlp_seed": 0,
        "mlp_name": "x",
        "weights": [[[1.0] * width] * width] * depth,
    }
    mlp = MLP.from_row(row)
    assert mlp.width == width
    assert mlp.depth == depth


def test_mlp_from_row_validates_shape():
    """Rows with malformed weights should raise via .validate()."""
    row = {
        "mlp_seed": 0,
        "mlp_name": "x",
        "weights": [[[1.0, 2.0], [3.0, 4.0, 5.0]]],  # ragged inner row
    }
    with pytest.raises(ValueError):
        MLP.from_row(row)


def test_mlp_from_row_pins_weights_to_float32() -> None:
    """Weights are float32, whatever dtype the row arrives in.

    Regression guard for a silent read-path upcast. The parquet stores float32
    (Arrow ``float``), but ``datasets`` returns list columns as nested PYTHON lists
    unless a format is set, and Python floats are C doubles — so building weights
    straight off an unformatted row yielded float64. Nothing failed; estimators were
    simply billed at flopscope's 2x float64 rate on everything touching the weights,
    and resident weight memory doubled.

    A plain ``list`` row is exactly the shape that produced the bug, so it is the
    case worth pinning.
    """
    row = {"weights": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], "mlp_seed": 7}
    mlp = MLP.from_row(row)
    assert all(w.dtype == fnp.float32 for w in mlp.weights)


def test_mlp_from_row_does_not_upcast_float32_input() -> None:
    """A float32 row stays float32 — the pin narrows dtype, it never widens it."""
    w = np.arange(4, dtype=np.float32).reshape(2, 2)
    mlp = MLP.from_row({"weights": [w], "mlp_seed": 0})
    assert mlp.weights[0].dtype == fnp.float32


def test_sample_mlp_weights_are_float32() -> None:
    """The generation path agrees with the read path, so a locally generated suite
    and a baked one bill identically."""
    from whestbench.generation import sample_mlp

    assert sample_mlp(width=4, depth=2).weights[0].dtype == fnp.float32
