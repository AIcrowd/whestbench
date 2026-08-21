import numpy as np
import pytest
from datasets import Dataset

from whestbench.domain import MLP
from whestbench.scoring import ContestSpec, make_contest_from_dataset
from whestbench.seeds import derive_estimator_seed, derive_estimator_seed_v4, generate_salt


def _ds(n=2, width=4, depth=2):
    rows = [
        {
            "mlp_id": i,
            "mlp_name": f"m-{i}",
            "mlp_seed": i * 1000 + 1,
            "weights": np.zeros((depth, width, width), dtype=np.float32).tolist(),
            "all_layer_means": np.zeros((depth, width), dtype=np.float32).tolist(),
            "final_means": np.zeros(width, dtype=np.float32).tolist(),
            "avg_variance": 0.1,
            "sampling_budget_breakdown": '{"flop_budget":0,"flops_used":0,"flops_remaining":0,"wall_time_s":0.0,"flopscope_backend_time_s":0.0,"flopscope_overhead_time_s":0.0,"residual_wall_time_s":0.0}',
        }
        for i in range(n)
    ]
    return Dataset.from_list(rows)


def _spec(n=2, width=4, depth=2):
    return ContestSpec(
        width=width, depth=depth, n_mlps=n, flop_budget=10**9, ground_truth_samples=8
    )


def test_unregistered_dataset_without_explicit_protocol_raises():
    with pytest.raises(ValueError, match="seed_protocol"):
        make_contest_from_dataset(_spec(), _ds(), n_mlps=2)


def test_explicit_protocol_3_0_uses_derived_seed():
    cd = make_contest_from_dataset(_spec(), _ds(), n_mlps=2, seed_protocol_version="3.0")
    assert cd.mlps[0].seed == derive_estimator_seed(1)  # mlp_seed=1
    assert cd.mlps[1].seed == derive_estimator_seed(1001)


def test_explicit_protocol_2_0_uses_raw_seed():
    cd = make_contest_from_dataset(_spec(), _ds(), n_mlps=2, seed_protocol_version="2.0")
    assert cd.mlps[0].seed == 1
    assert cd.mlps[1].seed == 1001


def test_explicit_protocol_invalid_version_string_raises():
    # A typo like "3" must not silently fall through to raw (2.0) seed handling —
    # that is the exact silent-wrong-score class the explicit path is meant to make loud.
    with pytest.raises(ValueError, match="unsupported seed_protocol_version"):
        make_contest_from_dataset(_spec(), _ds(), n_mlps=2, seed_protocol_version="3")


def test_explicit_protocol_non_string_raises():
    # e.g. accidentally passing the whole `seed_protocol` object from metadata.json.
    # The wrong type is the point of the test, so the static arg-type error is expected.
    with pytest.raises(ValueError, match="unsupported seed_protocol_version"):
        make_contest_from_dataset(
            _spec(),
            _ds(),
            n_mlps=2,
            seed_protocol_version={"version": "3.0"},  # type: ignore[arg-type]
        )


def test_from_row_rejects_unsupported_version():
    # Validation lives in the interpreter, so every caller (iter_mlps/load_mlp/direct) is guarded.
    row = _ds(n=1)[0]
    with pytest.raises(ValueError, match="unsupported seed_protocol_version"):
        MLP.from_row(row, seed_protocol_version="v3")


# --- protocol 4.0 needs a salt, and unregistered datasets need a way in ------


def test_protocol_4_0_on_an_unregistered_dataset_requires_an_explicit_salt():
    """Without the escape hatch a 4.0 dataset is simply unscoreable here.

    ``_METADATA_BY_DS`` is keyed on object identity and populated only by
    ``load_dataset``, so a concatenated, sliced or hand-built Dataset has no
    registered metadata to resolve a salt from. ``seed_protocol_version=`` alone
    is therefore not enough for 4.0, the way it is for 2.0 and 3.0.
    """
    with pytest.raises(ValueError, match="salt"):
        make_contest_from_dataset(_spec(), _ds(), n_mlps=2, seed_protocol_version="4.0")


def test_explicit_salt_makes_an_unregistered_4_0_dataset_scoreable():
    salt = generate_salt()
    cd = make_contest_from_dataset(
        _spec(), _ds(), n_mlps=2, seed_protocol_version="4.0", seed_salt=salt
    )
    # mlp_seed values are i*1000+1, i.e. 1 and 1001.
    assert [m.seed for m in cd.mlps] == [
        derive_estimator_seed_v4(1, salt=salt),
        derive_estimator_seed_v4(1001, salt=salt),
    ]


def test_an_explicit_salt_may_be_hex():
    # Mirrors the metadata form, so an operator can paste the recorded value.
    salt = generate_salt()
    as_bytes = make_contest_from_dataset(
        _spec(), _ds(), n_mlps=2, seed_protocol_version="4.0", seed_salt=salt
    )
    as_hex = make_contest_from_dataset(
        _spec(), _ds(), n_mlps=2, seed_protocol_version="4.0", seed_salt=salt.hex()
    )
    assert [m.seed for m in as_bytes.mlps] == [m.seed for m in as_hex.mlps]


def test_the_wrong_salt_yields_different_seeds_rather_than_failing_silently():
    # There is no digest to check against on an unregistered dataset, so the
    # caller owns salt correctness here. Documented so the difference from the
    # registered path (which verifies salt_digest) is explicit.
    a = make_contest_from_dataset(
        _spec(), _ds(), n_mlps=2, seed_protocol_version="4.0", seed_salt=generate_salt()
    )
    b = make_contest_from_dataset(
        _spec(), _ds(), n_mlps=2, seed_protocol_version="4.0", seed_salt=generate_salt()
    )
    assert [m.seed for m in a.mlps] != [m.seed for m in b.mlps]


def test_a_salt_is_ignored_for_pre_4_0_protocols():
    # Harmless: 3.0 does not consult it, so passing one cannot change a score.
    with_salt = make_contest_from_dataset(
        _spec(), _ds(), n_mlps=2, seed_protocol_version="3.0", seed_salt=generate_salt()
    )
    without = make_contest_from_dataset(_spec(), _ds(), n_mlps=2, seed_protocol_version="3.0")
    assert [m.seed for m in with_salt.mlps] == [m.seed for m in without.mlps]
