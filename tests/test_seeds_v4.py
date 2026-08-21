"""Seed-protocol 4.0: derivation, salt resolution, and the no-re-bake invariant."""

from __future__ import annotations

import os

import pytest

from whestbench.dataset_io import SEED_PROTOCOL_VERSION_V3, SEED_PROTOCOL_VERSION_V4
from whestbench.seeds import (
    SALT_NBYTES,
    SEED_SALT_ENV_VAR,
    bake_seed_values,
    build_seed_protocol_metadata,
    derive_estimator_seed_v4,
    derive_name_seed_v4,
    derive_seed_streams,
    generate_salt,
    resolve_bake_salt,
    resolve_salt,
    salt_digest,
)

# A fixed (seed, salt) anchor. Pins the wire format -- BLAKE2b personalisation
# strings, key handling, byte order, and the uint32 mask -- so an accidental
# change to any of them fails here instead of silently renaming every MLP in
# every 4.0 dataset. Deliberately synthetic: test fixtures must never embed a
# real dataset's input seed.
_ANCHOR_SEED = 6_004_799_503_160_661
_ANCHOR_SALT = bytes(range(32))


def test_anchor_estimator_seed_is_stable():
    assert derive_estimator_seed_v4(_ANCHOR_SEED, salt=_ANCHOR_SALT) == 1_914_761_819


def test_anchor_name_seed_is_stable():
    assert derive_name_seed_v4(_ANCHOR_SEED, salt=_ANCHOR_SALT) == 1_338_665_652_895_792_380


def test_estimator_seed_stays_in_the_uint32_range_protocol_3_produced():
    # Participant estimator code calls np.random.seed(mlp.seed), which rejects
    # anything >= 2**32. Widening this silently breaks submissions.
    salt = generate_salt()
    for s in (0, 1, (1 << 63) - 1, _ANCHOR_SEED):
        assert 0 <= derive_estimator_seed_v4(s, salt=salt) < 2**32


def test_estimator_and_name_seeds_are_domain_separated():
    # Same input, same key: if these collided, holding one would yield the other.
    salt = generate_salt()
    assert derive_estimator_seed_v4(_ANCHOR_SEED, salt=salt) != derive_name_seed_v4(
        _ANCHOR_SEED, salt=salt
    )


def test_derivation_depends_on_the_key():
    a, b = generate_salt(), generate_salt()
    assert derive_estimator_seed_v4(_ANCHOR_SEED, salt=a) != derive_estimator_seed_v4(
        _ANCHOR_SEED, salt=b
    )


def test_derivation_is_deterministic():
    salt = generate_salt()
    assert derive_estimator_seed_v4(_ANCHOR_SEED, salt=salt) == derive_estimator_seed_v4(
        _ANCHOR_SEED, salt=salt
    )


def test_generated_salt_is_full_width():
    # salt_digest is a plain SHA-256 recorded in metadata; a short salt would
    # make that digest an offline target.
    assert len(generate_salt()) == SALT_NBYTES == 32
    assert generate_salt() != generate_salt()


# --- The invariant the whole no-re-bake migration rests on --------------------


@pytest.mark.parametrize("input_seed", [0, 1, 1001, _ANCHOR_SEED, (1 << 63) - 1])
def test_data_streams_are_untouched_by_protocol_4(input_seed):
    """4.0 must not perturb the weight or ground-truth streams.

    A 3.0 dataset is migrated to 4.0 by relabelling metadata, WITHOUT
    regenerating weights or targets. That is only sound while both protocols
    read the same two substreams, so this pins it. If it ever fails, every
    already-baked dataset is silently invalidated.
    """
    weight_ss, sample_ss, _ = derive_seed_streams(input_seed)
    est, name = bake_seed_values(
        [input_seed], seed_protocol_version=SEED_PROTOCOL_VERSION_V4, salt=generate_salt()
    )
    again_weight_ss, again_sample_ss, _ = derive_seed_streams(input_seed)

    assert list(weight_ss.pool) == list(again_weight_ss.pool)
    assert list(sample_ss.pool) == list(again_sample_ss.pool)
    assert est and name  # the 4.0 values exist and did not touch the streams above


def test_bake_seed_values_keeps_name_tied_to_seed_under_protocol_3():
    # 3.0's documented behaviour: mlp_name is seeded from the estimator seed.
    seeds = [1001, 2002, 3003]
    est, name = bake_seed_values(seeds, seed_protocol_version=SEED_PROTOCOL_VERSION_V3)
    assert est == name == [derive_seed_streams(s)[2] for s in seeds]


def test_bake_seed_values_requires_a_salt_for_protocol_4():
    with pytest.raises(ValueError, match="requires a salt"):
        bake_seed_values([1001], seed_protocol_version=SEED_PROTOCOL_VERSION_V4)


# --- Salt resolution: every failure path raises, none defaults ---------------


def test_resolve_salt_reads_the_metadata_salt():
    salt = generate_salt()
    proto = {"salt_source": "metadata", "salt": salt.hex(), "salt_digest": salt_digest(salt)}
    assert resolve_salt(proto) == salt


def test_resolve_salt_rejects_missing_metadata_salt():
    with pytest.raises(ValueError, match="missing the 'salt' field"):
        resolve_salt({"salt_source": "metadata"})


def test_resolve_salt_rejects_a_salt_that_does_not_match_its_digest():
    # The wrong salt derives different seeds and names. It must not be silent.
    proto = {"salt": generate_salt().hex(), "salt_digest": salt_digest(generate_salt())}
    with pytest.raises(ValueError, match="does not match salt_digest"):
        resolve_salt(proto)


def test_resolve_salt_rejects_non_hex():
    with pytest.raises(ValueError, match="not valid hex"):
        resolve_salt({"salt": "nothex!!"})


def test_resolve_salt_rejects_an_unknown_salt_source():
    with pytest.raises(ValueError, match="unsupported seed_protocol salt_source"):
        resolve_salt({"salt_source": "s3"})


def test_env_salt_source_hard_errors_when_the_var_is_unset(monkeypatch):
    """The anti-fallback test.

    A missing override must raise, never quietly fall back to a default salt --
    that would derive plausible-but-wrong seeds and names, and every score
    computed from them would be wrong with no signal.
    """
    monkeypatch.delenv(SEED_SALT_ENV_VAR, raising=False)
    salt = generate_salt()
    with pytest.raises(ValueError, match=SEED_SALT_ENV_VAR):
        resolve_salt({"salt_source": "env", "salt_digest": salt_digest(salt)})


def test_env_salt_source_reads_the_variable(monkeypatch):
    salt = generate_salt()
    monkeypatch.setenv(SEED_SALT_ENV_VAR, salt.hex())
    proto = {"salt_source": "env", "salt_digest": salt_digest(salt)}
    assert resolve_salt(proto) == salt


def test_env_salt_source_still_verifies_the_digest(monkeypatch):
    monkeypatch.setenv(SEED_SALT_ENV_VAR, generate_salt().hex())
    proto = {"salt_source": "env", "salt_digest": salt_digest(generate_salt())}
    with pytest.raises(ValueError, match="does not match salt_digest"):
        resolve_salt(proto)


def test_env_var_uses_the_repo_prefix():
    assert SEED_SALT_ENV_VAR.startswith("WHEST_")


# --- Bake-time salt handling -------------------------------------------------


def test_sliced_protocol_4_bake_refuses_to_invent_a_salt():
    """Each fleet worker is a separate process.

    Auto-generating per worker would give each shard a different mlp_name
    column, and the shards would merge without complaint.
    """
    with pytest.raises(ValueError, match="must be given an explicit seed_salt"):
        resolve_bake_salt(
            seed_protocol_version=SEED_PROTOCOL_VERSION_V4, seed_salt=None, is_partial=True
        )


def test_whole_dataset_protocol_4_bake_generates_a_salt():
    salt = resolve_bake_salt(
        seed_protocol_version=SEED_PROTOCOL_VERSION_V4, seed_salt=None, is_partial=False
    )
    assert salt is not None and len(salt) == SALT_NBYTES


def test_sliced_protocol_4_bake_accepts_an_explicit_salt():
    salt = generate_salt()
    assert (
        resolve_bake_salt(
            seed_protocol_version=SEED_PROTOCOL_VERSION_V4, seed_salt=salt, is_partial=True
        )
        == salt
    )


def test_salt_is_rejected_for_pre_4_protocols():
    with pytest.raises(ValueError, match="only meaningful for seed_protocol"):
        resolve_bake_salt(
            seed_protocol_version=SEED_PROTOCOL_VERSION_V3,
            seed_salt=generate_salt(),
            is_partial=False,
        )


# --- Metadata block ----------------------------------------------------------


def test_metadata_block_records_the_salt_by_default():
    salt = generate_salt()
    block = build_seed_protocol_metadata(seed_protocol_version=SEED_PROTOCOL_VERSION_V4, salt=salt)
    assert block["version"] == SEED_PROTOCOL_VERSION_V4
    assert block["salt_source"] == "metadata"
    assert block["salt"] == salt.hex()
    assert block["salt_digest"] == salt_digest(salt)


def test_withheld_salt_is_absent_from_metadata_but_its_digest_is_not():
    salt = generate_salt()
    block = build_seed_protocol_metadata(
        seed_protocol_version=SEED_PROTOCOL_VERSION_V4,
        salt=salt,
        salt_id="eval-2026",
        withhold_salt=True,
    )
    assert "salt" not in block, "withholding the salt must not ship it anyway"
    assert block["salt_source"] == "env"
    assert block["salt_id"] == "eval-2026"
    assert block["salt_digest"] == salt_digest(salt)


def test_withholding_the_salt_requires_a_salt_id():
    # Without one, an operator cannot tell which salt to export.
    with pytest.raises(ValueError, match="requires salt_id"):
        build_seed_protocol_metadata(
            seed_protocol_version=SEED_PROTOCOL_VERSION_V4,
            salt=generate_salt(),
            withhold_salt=True,
        )


def test_protocol_3_metadata_block_carries_no_salt_fields():
    block = build_seed_protocol_metadata(seed_protocol_version=SEED_PROTOCOL_VERSION_V3)
    assert block["version"] == SEED_PROTOCOL_VERSION_V3
    assert "salt" not in block and "salt_digest" not in block


def test_metadata_block_rejects_an_unbakeable_protocol():
    with pytest.raises(ValueError, match="cannot bake seed_protocol_version"):
        build_seed_protocol_metadata(seed_protocol_version="2.0")


def test_no_default_salt_constant_exists():
    """Guards the rule that makes a missing salt a hard error.

    No code path may construct salt bytes: every branch either returns bytes
    that came from metadata, the environment, or an explicit argument, or it
    raises. A "convenience" default added later would reintroduce silent
    wrong-seed derivation, so fail loudly if one appears.
    """
    import whestbench.seeds as seeds_mod

    for name in dir(seeds_mod):
        if "SALT" in name.upper() and name != "SALT_NBYTES":
            value = getattr(seeds_mod, name)
            assert not isinstance(value, bytes), (
                f"whestbench.seeds.{name} looks like a default salt constant; "
                "protocol 4.0 must have no fallback salt."
            )
    assert os.environ.get("__never_set__") is None
