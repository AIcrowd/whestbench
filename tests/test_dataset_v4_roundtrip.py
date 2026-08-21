"""Seed-protocol 4.0 end to end: bake, validate, load, and read back."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pytest

import whestbench
from whestbench.dataset import create_dataset, iter_mlps, load_dataset
from whestbench.dataset_io import (
    METADATA_FILE,
    SEED_PROTOCOL_NAME_V4,
    SEED_PROTOCOL_VERSION_V4,
    InvalidDatasetError,
    validate_metadata,
)
from whestbench.domain import MLP
from whestbench.naming import assign_unique_names
from whestbench.seeds import (
    SEED_SALT_ENV_VAR,
    derive_estimator_seed_v4,
    derive_name_seed_v4,
    generate_salt,
    resolve_salt,
    salt_digest,
)

_SEEDS = [1001, 2002, 3003, 4004]
_BAKE: dict[str, Any] = dict(
    n_mlps=4, n_samples=64, width=4, depth=2, mlp_seeds=_SEEDS, split="public"
)


def _bake(tmp_path, name, **kw):
    return create_dataset(output_path=tmp_path / name, **_BAKE, **kw)


def test_protocol_4_bake_leaves_weights_and_ground_truth_identical_to_protocol_3(tmp_path):
    """The claim the whole no-re-bake migration rests on.

    An existing 3.0 dataset is moved to 4.0 by relabelling metadata and
    rewriting mlp_name -- weights and targets are never regenerated. This proves
    the two protocols really do produce the same data columns for the same
    seeds. If it fails, migrating an existing dataset silently corrupts it.
    """
    _bake(tmp_path, "v3")
    _bake(tmp_path, "v4", seed_protocol_version=SEED_PROTOCOL_VERSION_V4)
    d3 = load_dataset(str(tmp_path / "v3"), split="public")
    d4 = load_dataset(str(tmp_path / "v4"), split="public")

    for column in ("weights", "all_layer_means", "final_means", "avg_variance", "mlp_seed"):
        assert np.array_equal(np.asarray(d3[column]), np.asarray(d4[column])), (
            f"{column} differs between protocol 3.0 and 4.0; migrating an existing "
            f"dataset by relabelling metadata would corrupt it"
        )


def test_protocol_4_changes_the_participant_facing_values(tmp_path):
    # The flip side of the test above: identifiers must NOT carry over, or 4.0
    # would inherit 3.0's derivation.
    _bake(tmp_path, "v3")
    _bake(tmp_path, "v4", seed_protocol_version=SEED_PROTOCOL_VERSION_V4)
    d3 = load_dataset(str(tmp_path / "v3"), split="public")
    d4 = load_dataset(str(tmp_path / "v4"), split="public")

    assert list(d3["mlp_name"]) != list(d4["mlp_name"])
    assert [m.seed for m in iter_mlps(d3)] != [m.seed for m in iter_mlps(d4)]


def test_round_trip_seeds_and_names_match_the_derivation(tmp_path):
    _bake(tmp_path, "v4", seed_protocol_version=SEED_PROTOCOL_VERSION_V4)
    ds = load_dataset(str(tmp_path / "v4"), split="public")
    salt = resolve_salt(whestbench.metadata(ds)["seed_protocol"])

    assert [m.seed for m in iter_mlps(ds)] == [
        derive_estimator_seed_v4(s, salt=salt) for s in _SEEDS
    ]
    assert list(ds["mlp_name"]) == assign_unique_names(
        [derive_name_seed_v4(s, salt=salt) for s in _SEEDS]
    )


def test_baked_estimator_seeds_stay_in_the_uint32_range(tmp_path):
    # np.random.seed(mlp.seed) must keep working for existing submissions.
    _bake(tmp_path, "v4", seed_protocol_version=SEED_PROTOCOL_VERSION_V4)
    ds = load_dataset(str(tmp_path / "v4"), split="public")
    for mlp in iter_mlps(ds):
        assert 0 <= mlp.seed < 2**32
        np.random.seed(mlp.seed)  # would raise if the range widened


def test_bake_records_a_usable_seed_protocol_block(tmp_path):
    path = _bake(tmp_path, "v4", seed_protocol_version=SEED_PROTOCOL_VERSION_V4)
    block = json.loads((path / METADATA_FILE).read_text())["seed_protocol"]

    assert block["name"] == SEED_PROTOCOL_NAME_V4
    assert block["version"] == SEED_PROTOCOL_VERSION_V4
    assert block["salt_source"] == "metadata"
    assert salt_digest(bytes.fromhex(block["salt"])) == block["salt_digest"]


def test_withheld_salt_bake_omits_the_salt_and_needs_the_env_var(tmp_path, monkeypatch):
    path = _bake(
        tmp_path,
        "v4env",
        seed_protocol_version=SEED_PROTOCOL_VERSION_V4,
        salt_id="eval-2026",
        withhold_salt=True,
    )
    block = json.loads((path / METADATA_FILE).read_text())["seed_protocol"]
    assert "salt" not in block
    assert block["salt_source"] == "env"

    monkeypatch.delenv(SEED_SALT_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match=SEED_SALT_ENV_VAR):
        list(iter_mlps(load_dataset(str(path), split="public")))


def test_sliced_protocol_4_bake_without_an_explicit_salt_is_refused(tmp_path):
    with pytest.raises(ValueError, match="must be given an explicit seed_salt"):
        create_dataset(
            output_path=tmp_path / "partial",
            mlp_range=(0, 2),
            seed_protocol_version=SEED_PROTOCOL_VERSION_V4,
            **_BAKE,
        )


def test_slices_sharing_a_salt_agree_with_the_whole_dataset_bake(tmp_path):
    """Sliced-bake equivalence, the property fleet bakes rely on.

    Names are assigned across ALL n_mlps then sliced, so a worker's shard must
    carry the same names as the corresponding slice of a single-host bake --
    otherwise merged shards disagree about who is who.

    Reads the partial's parquet directly: ``load_dataset`` refuses partials
    until they are merged.
    """
    from datasets import Dataset

    salt = generate_salt()
    whole = _bake(tmp_path, "whole", seed_protocol_version=SEED_PROTOCOL_VERSION_V4, seed_salt=salt)
    part = create_dataset(
        output_path=tmp_path / "part",
        mlp_range=(1, 3),
        seed_protocol_version=SEED_PROTOCOL_VERSION_V4,
        seed_salt=salt,
        **_BAKE,
    )
    whole_names = list(load_dataset(str(whole), split="public")["mlp_name"])
    part_ds = Dataset.from_parquet(str(next((part / "data").glob("*.parquet"))))
    assert list(part_ds["mlp_name"]) == whole_names[1:3]  # pyright: ignore[reportIndexIssue]


# --- the salt must never reach the dataset card ------------------------------


@pytest.mark.parametrize("withhold", [False, True])
def test_dataset_card_never_contains_the_raw_salt(tmp_path, withhold):
    """The card is the most public artifact a dataset has.

    Protocol 4.0's whole guarantee is that the salt stays secret, so a template
    edit that interpolates ``seed_protocol.salt`` -- easy to do by accident when
    adding a re-bake recipe -- would silently void it for every dataset baked
    afterwards. Checked in BOTH salt modes: the metadata-salt case is the one
    that can leak, and the withheld case must not regress into embedding it.
    """
    extra = {"salt_id": "eval-2026", "withhold_salt": True} if withhold else {}
    path = _bake(
        tmp_path,
        f"card-{withhold}",
        seed_protocol_version=SEED_PROTOCOL_VERSION_V4,
        **extra,
    )
    salt_hex = json.loads((path / METADATA_FILE).read_text())["seed_protocol"].get("salt")
    card = (path / "README.md").read_text()

    if salt_hex is not None:
        assert salt_hex not in card, "the dataset card leaks the raw salt"
        assert bytes.fromhex(salt_hex).hex() not in card.lower()
    # Nothing salt-shaped at all: no bare 64-hex-char run anywhere in the card.
    assert not re.search(r"\b[0-9a-fA-F]{64}\b", card), (
        "the dataset card contains a 64-hex-character token; if that is the salt "
        "(or its preimage) protocol 4.0's guarantee is void"
    )


def test_dataset_card_renders_the_protocol_4_section(tmp_path):
    # Both existing branches match on protocol name, so a missing 4.0 branch
    # drops the Reproducibility section silently rather than failing.
    path = _bake(tmp_path, "card-render", seed_protocol_version=SEED_PROTOCOL_VERSION_V4)
    card = (path / "README.md").read_text()
    assert "seed_protocol 4.0 (keyed per-MLP seeds)" in card
    assert "seed_protocol 4.0" in card


def test_withheld_salt_card_tells_the_reader_which_salt_to_export(tmp_path):
    path = _bake(
        tmp_path,
        "card-env",
        seed_protocol_version=SEED_PROTOCOL_VERSION_V4,
        salt_id="eval-2026",
        withhold_salt=True,
    )
    card = (path / "README.md").read_text()
    assert SEED_SALT_ENV_VAR in card
    assert "eval-2026" in card


def test_protocol_4_card_describes_the_readers_own_protocol(tmp_path):
    # The mlp_seed / mlp_name schema rows must not silently describe 2.0/3.0
    # semantics to a 4.0 reader.
    v4 = _bake(tmp_path, "schema-v4", seed_protocol_version=SEED_PROTOCOL_VERSION_V4)
    v3 = _bake(tmp_path, "schema-v3")
    card4 = (v4 / "README.md").read_text()
    card3 = (v3 / "README.md").read_text()

    assert "This dataset uses seed_protocol 4.0" in card4
    assert "This dataset uses seed_protocol 4.0" not in card3
    assert "carries no information beyond `mlp_seed`" in card3
    assert "carries no information beyond `mlp_seed`" not in card4


# --- metadata validation -----------------------------------------------------


def _v4_md(**overrides):
    salt = generate_salt()
    proto = {
        "name": SEED_PROTOCOL_NAME_V4,
        "version": SEED_PROTOCOL_VERSION_V4,
        "salt_source": "metadata",
        "salt": salt.hex(),
        "salt_digest": salt_digest(salt),
    }
    proto.update(overrides)
    return {
        "schema_version": "3.0",
        "format": "hf-datasets-parquet",
        "seed_protocol": proto,
        "n_mlps": 4,
        "n_samples": 64,
        "width": 4,
        "depth": 2,
    }


def test_validate_accepts_protocol_4_metadata():
    validate_metadata(_v4_md())


def test_validate_rejects_protocol_4_with_a_top_level_seed():
    # 4.0 shares 3.0's shape: per-MLP seeds live in the parquet, not metadata.
    md = _v4_md()
    md["seed"] = 42
    with pytest.raises(InvalidDatasetError, match=r"seed_protocol.+4\.0.+top-level.+seed"):
        validate_metadata(md)


def test_validate_rejects_protocol_4_without_a_salt():
    with pytest.raises(InvalidDatasetError, match="missing the 'salt' field"):
        validate_metadata(_v4_md(salt=None))


def test_validate_rejects_a_shipped_salt_when_the_source_is_env():
    # Withholding the salt but shipping it anyway defeats the point.
    with pytest.raises(InvalidDatasetError, match="a 'salt' field is also present"):
        validate_metadata(_v4_md(salt_source="env", salt_id="x"))


def test_validate_rejects_env_source_without_a_digest():
    md = _v4_md(salt_source="env", salt_id="x")
    del md["seed_protocol"]["salt"]
    del md["seed_protocol"]["salt_digest"]
    with pytest.raises(InvalidDatasetError, match="requires 'salt_digest'"):
        validate_metadata(md)


def test_validate_rejects_a_non_hex_salt():
    with pytest.raises(InvalidDatasetError, match="not valid hex"):
        validate_metadata(_v4_md(salt="zzzz"))


# --- from_row guardrails -----------------------------------------------------


def test_from_row_refuses_protocol_4_without_a_salt():
    """No unsalted fallback.

    Deriving without the salt would yield a plausible-looking seed that differs
    from the one the dataset was baked with, and every score would be silently
    wrong.
    """
    row = {"weights": [np.zeros((2, 2), dtype=np.float32)], "mlp_seed": 1001, "mlp_name": "x"}
    with pytest.raises(ValueError, match="requires seed_salt"):
        MLP.from_row(row, seed_protocol_version=SEED_PROTOCOL_VERSION_V4)


def test_from_row_uses_the_salt_when_given():
    salt = generate_salt()
    row = {"weights": [np.zeros((2, 2), dtype=np.float32)], "mlp_seed": 1001, "mlp_name": "x"}
    mlp = MLP.from_row(row, seed_protocol_version=SEED_PROTOCOL_VERSION_V4, seed_salt=salt)
    assert mlp.seed == derive_estimator_seed_v4(1001, salt=salt)


@pytest.mark.parametrize("bad", ["4", "v4", "4.0.0"])
def test_from_row_still_rejects_near_miss_versions(bad):
    row = {"weights": [np.zeros((2, 2), dtype=np.float32)], "mlp_seed": 1001}
    with pytest.raises(ValueError, match="unsupported seed_protocol_version"):
        MLP.from_row(row, seed_protocol_version=bad)
