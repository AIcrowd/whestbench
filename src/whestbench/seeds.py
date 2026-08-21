"""Single authority for the per-MLP seed derivation (seed-protocols 3.0 and 4.0).

A dataset row stores one ROOT input seed per MLP. Both protocols generate the
weights and the ground truth from the same two substreams of that root:

    ss = SeedSequence(input_seed).spawn(3)
    ss[0] -> weight stream      (MLP weight matrices)
    ss[1] -> sample stream      (ground-truth Monte-Carlo draws)

They differ only in what the PARTICIPANT is handed.

Protocol 3.0 hands over the third substream, ``ss[2].generate_state(1)[0]``.
Protocol 4.0 hands over a keyed BLAKE2b of ``input_seed`` under a per-dataset
secret, and does not use ``ss[2]`` at all.

Maintenance rules for 4.0. Each of these is load bearing; changing any of them
weakens the protocol, so do not "simplify" one without reading
``docs/reference/dataset-format.md`` and the seed-protocol design note first:

  * The derivation MUST consume ``input_seed`` itself, never a value narrowed
    from it. A narrowed input is enumerable; the 63-bit root is not.
  * The derivation MUST stay keyed. Unkeyed, it is computable by anyone holding
    a derived value, which is precisely the property 4.0 exists to remove.
  * The estimator seed and the name seed MUST stay domain-separated, so that
    holding one does not yield the other.
  * The key is a SECRET, not a public nonce. Do not publish it, log it, print
    it in a CLI, or render it into a dataset card. ``salt_source="env"`` exists
    for bakes that want it out of the shipped directory entirely.

``weight_ss`` and ``sample_ss`` are byte-identical under both protocols, which
is what lets a 3.0 dataset be migrated to 4.0 without re-baking. Do NOT route
the data streams through the keyed derivation: it would gain nothing and would
silently change the baked ground truth.

This module is the ONE place that computes the participant-facing seed; bake and
read both call it so the derivation cannot drift.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from typing import Any

import flopscope.numpy as fnp

# --- Protocol 4.0 derivation --------------------------------------------------

# Domain separation. The estimator seed and the name seed are derived from the
# same input seed under the same key, so they must not be the same function of
# it -- otherwise holding one yields the other. BLAKE2b caps ``person`` at 16
# bytes; these are part of the protocol, and changing one changes every derived
# value in every 4.0 dataset.
_PERSON_ESTIMATOR = b"whest-4.0-est"
_PERSON_NAME = b"whest-4.0-name"

# ``MLP.seed`` stays in the uint32 range that protocol 3.0 produced. This is a
# participant-facing compatibility contract, not a security choice: estimator
# code in the wild calls ``np.random.seed(mlp.seed)``, which rejects anything
# >= 2**32. Width does not affect key secrecy, and a narrower value leaves MORE
# input seeds consistent with it, so there is nothing to lose here.
_UINT32_MASK = (1 << 32) - 1
_INT63_MASK = (1 << 63) - 1

#: Environment variable carrying a hex salt when ``salt_source`` is ``"env"``.
#: Matches the repo's ``WHEST_`` prefix (see ``WHEST_MAX_THREADS``).
SEED_SALT_ENV_VAR = "WHEST_SEED_SALT"

#: Byte length of a generated protocol-4.0 salt. 256 bits, and load-bearing:
#: ``salt_digest`` is a plain SHA-256 of these bytes recorded in metadata, so a
#: short salt would make that digest an offline brute-force target.
SALT_NBYTES = 32


def generate_salt() -> bytes:
    """Return a fresh per-dataset protocol-4.0 salt."""
    return secrets.token_bytes(SALT_NBYTES)


def salt_digest(salt: bytes) -> str:
    """Return the SHA-256 hex digest recorded in metadata for salt verification."""
    return hashlib.sha256(salt).hexdigest()


def _keyed_digest(input_seed: int, *, salt: bytes, person: bytes) -> int:
    """Keyed BLAKE2b of a per-MLP input seed, as an int.

    ``input_seed`` MUST be the full 63-bit root; a value narrowed from it is
    enumerable and must never be passed here.

    Keyed BLAKE2b is a MAC, so this is a PRF of the input seed under ``salt``.
    ``hashlib.blake2b`` is built into CPython with no OpenSSL dependency, so 4.0
    datasets are readable on every supported interpreter.
    """
    return int.from_bytes(
        hashlib.blake2b(
            int(input_seed).to_bytes(8, "big"),
            digest_size=8,
            key=salt,
            person=person,
        ).digest(),
        "big",
    )


def derive_estimator_seed_v4(input_seed: int, *, salt: bytes) -> int:
    """The protocol-4.0 estimator seed (``MLP.seed``) for a per-MLP input seed.

    Masked to uint32 to match the range protocol 3.0 produced -- see
    ``_UINT32_MASK``. Participant estimator code sees no range change.
    """
    return _keyed_digest(input_seed, salt=salt, person=_PERSON_ESTIMATOR) & _UINT32_MASK


def derive_name_seed_v4(input_seed: int, *, salt: bytes) -> int:
    """The protocol-4.0 seed fed to ``naming.generate_mlp_name``.

    Domain-separated from :func:`derive_estimator_seed_v4`, so holding one does
    not yield the other. Not part of any participant-facing contract (only the
    resulting slug is), so it keeps the full int63 range.
    """
    return _keyed_digest(input_seed, salt=salt, person=_PERSON_NAME) & _INT63_MASK


def resolve_salt(seed_protocol: "dict | None") -> bytes:
    """Return the protocol-4.0 salt declared by a dataset's ``seed_protocol``.

    Two sources, and NEITHER falls back to a default. A silently-missing salt
    would produce wrong-but-plausible estimator seeds and names rather than an
    error, so every failure path here raises.

    ``salt_source="metadata"`` (default) reads the hex ``salt`` field.
    ``salt_source="env"`` reads ``WHEST_SEED_SALT`` instead, for eval bakes
    that withhold the salt from the shipped metadata.

    When ``salt_digest`` is present the resolved salt is verified against it, so
    supplying the wrong salt fails immediately instead of silently renaming
    every MLP.
    """
    proto = seed_protocol or {}
    source = proto.get("salt_source", "metadata")

    if source == "metadata":
        raw = proto.get("salt")
        if not raw:
            raise ValueError(
                "seed_protocol 4.0 metadata is missing the 'salt' field "
                "(salt_source='metadata'). The dataset cannot be read without it."
            )
        salt = _decode_salt(raw, origin="seed_protocol.salt")
    elif source == "env":
        raw = os.environ.get(SEED_SALT_ENV_VAR)
        if not raw:
            raise ValueError(
                f"seed_protocol 4.0 declares salt_source='env' but {SEED_SALT_ENV_VAR} "
                f"is not set. Export the dataset's salt as hex in {SEED_SALT_ENV_VAR}; "
                "there is deliberately no default, because falling back to one would "
                "produce wrong-but-plausible seeds and names instead of an error."
            )
        salt = _decode_salt(raw, origin=SEED_SALT_ENV_VAR)
    else:
        raise ValueError(
            f"unsupported seed_protocol salt_source {source!r}; expected 'metadata' or 'env'"
        )

    expected = proto.get("salt_digest")
    if expected:
        actual = salt_digest(salt)
        if actual != expected:
            raise ValueError(
                f"seed_protocol 4.0 salt does not match salt_digest "
                f"(got {actual[:16]}..., expected {str(expected)[:16]}...). "
                "This salt would derive different estimator seeds and MLP names than "
                "the dataset was baked with."
            )
    return salt


def _decode_salt(raw: "str | bytes", *, origin: str) -> bytes:
    if isinstance(raw, bytes):
        return raw
    try:
        return bytes.fromhex(str(raw))
    except ValueError as exc:
        raise ValueError(f"{origin} is not valid hex: {exc}") from exc


# --- Protocol 3.0 (unchanged; 3.0 datasets stay loadable) ---------------------


def build_seed_protocol_metadata(
    *,
    seed_protocol_version: str,
    salt: "bytes | None" = None,
    salt_id: "str | None" = None,
    withhold_salt: bool = False,
) -> "dict[str, Any]":
    """Build the ``seed_protocol`` block a bake writes into ``metadata.json``.

    Shared by both backends so the CPU and GPU bakes cannot describe themselves
    differently.

    For 4.0, ``withhold_salt=True`` records ``salt_source="env"`` and omits the
    salt itself, for eval bakes that keep it out of the shipped directory. The
    digest is always recorded, so a wrong salt is caught on load rather than
    silently deriving different seeds and names than the bake used.
    """
    from .dataset_io import (
        SEED_PROTOCOL_NAME_V3,
        SEED_PROTOCOL_NAME_V4,
        SEED_PROTOCOL_VERSION_V3,
        SEED_PROTOCOL_VERSION_V4,
    )

    if seed_protocol_version == SEED_PROTOCOL_VERSION_V3:
        return {"name": SEED_PROTOCOL_NAME_V3, "version": SEED_PROTOCOL_VERSION_V3}
    if seed_protocol_version != SEED_PROTOCOL_VERSION_V4:
        raise ValueError(
            f"cannot bake seed_protocol_version {seed_protocol_version!r}; expected "
            f"{SEED_PROTOCOL_VERSION_V3!r} or {SEED_PROTOCOL_VERSION_V4!r}."
        )
    if salt is None:
        raise ValueError("seed_protocol 4.0 metadata requires the bake salt")
    if withhold_salt and not salt_id:
        raise ValueError(
            "withhold_salt=True requires salt_id, so an operator can tell which salt "
            f"to export as {SEED_SALT_ENV_VAR} when reading this dataset."
        )

    block: "dict[str, Any]" = {
        "name": SEED_PROTOCOL_NAME_V4,
        "version": SEED_PROTOCOL_VERSION_V4,
        "kdf": "blake2b-keyed",
        "salt_source": "env" if withhold_salt else "metadata",
        "salt_digest": salt_digest(salt),
    }
    if salt_id:
        block["salt_id"] = salt_id
    if not withhold_salt:
        block["salt"] = salt.hex()
    return block


def resolve_bake_salt(
    *,
    seed_protocol_version: str,
    seed_salt: "bytes | str | None",
    is_partial: bool,
) -> "bytes | None":
    """Resolve the salt a bake should use, or ``None`` for pre-4.0 protocols.

    Auto-generation is only safe for a whole-dataset bake. A sliced bake runs
    once per fleet worker in a separate process, so auto-generating would give
    each worker a different salt and therefore a different ``mlp_name`` column;
    the shards would then merge without complaint into a dataset whose names are
    internally inconsistent. So a sliced 4.0 bake must be handed an explicit
    salt, exactly as it must already be handed an explicit ``mlp_seeds`` list.
    """
    from .dataset_io import SEED_PROTOCOL_VERSION_V4

    if seed_protocol_version != SEED_PROTOCOL_VERSION_V4:
        if seed_salt is not None:
            raise ValueError(
                f"seed_salt is only meaningful for seed_protocol "
                f"{SEED_PROTOCOL_VERSION_V4}; got seed_protocol_version="
                f"{seed_protocol_version!r}."
            )
        return None

    if seed_salt is not None:
        return _decode_salt(seed_salt, origin="seed_salt")
    if is_partial:
        raise ValueError(
            f"a sliced seed_protocol {SEED_PROTOCOL_VERSION_V4} bake (mlp_range set) must be "
            "given an explicit seed_salt: auto-generating one per worker would give each "
            "shard a different mlp_name column, and they would merge without complaint. "
            "Generate the salt once and pass the same value to every worker, as you "
            "already do for mlp_seeds."
        )
    return generate_salt()


def bake_seed_values(
    mlp_seeds: "list[int]",
    *,
    seed_protocol_version: str,
    salt: "bytes | None" = None,
) -> "tuple[list[int], list[int]]":
    """Return ``(estimator_seeds, name_seeds)`` for a whole dataset at bake time.

    One helper for both backends. ``dataset.py`` and ``dataset_torch.py`` derive
    these identically on purpose -- the CPU and GPU bakes must produce the same
    ``mlp_name`` column for the same seeds -- so the logic lives here rather than
    being mirrored in two places that can drift apart.

    Under 3.0 the two lists are identical: the name has always been seeded from
    the estimator seed. Under 4.0 they are domain-separated, so a name no longer
    yields the seed it sits beside.

    Callers pass the FULL ``mlp_seeds`` list even when baking a slice, then slice
    the returned names -- ``assign_unique_names`` resolves collisions by order of
    first appearance across the whole list, so a per-slice call would produce
    different suffixes on different workers.
    """
    from .dataset_io import SEED_PROTOCOL_VERSION_V4

    if seed_protocol_version == SEED_PROTOCOL_VERSION_V4:
        if salt is None:
            raise ValueError("baking seed_protocol 4.0 requires a salt")
        estimator_seeds = [derive_estimator_seed_v4(s, salt=salt) for s in mlp_seeds]
        name_seeds = [derive_name_seed_v4(s, salt=salt) for s in mlp_seeds]
        return estimator_seeds, name_seeds

    estimator_seeds = [derive_seed_streams(s)[2] for s in mlp_seeds]
    return estimator_seeds, list(estimator_seeds)


def derive_seed_streams(
    input_seed: int,
) -> "tuple[fnp.random.SeedSequence, fnp.random.SeedSequence, int]":
    """Return ``(weight_ss, sample_ss, estimator_seed)`` for a per-MLP input seed.

    ``weight_ss`` and ``sample_ss`` are the streams BOTH protocols use; only the
    third return value is 3.0-specific. Callers baking 4.0 take the first two and
    derive the estimator seed via :func:`derive_estimator_seed_v4` instead.
    """
    weight_ss, sample_ss, estimator_ss = fnp.random.SeedSequence(int(input_seed)).spawn(3)
    estimator_seed = int(estimator_ss.generate_state(1)[0])
    return weight_ss, sample_ss, estimator_seed


def derive_estimator_seed(input_seed: int) -> int:
    """The protocol-3.0 estimator seed (``MLP.seed``) for a per-MLP input seed."""
    return derive_seed_streams(input_seed)[2]
