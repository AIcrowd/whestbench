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
