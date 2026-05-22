import flopscope.numpy as fnp
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
