"""Single authority for the per-MLP seed derivation (seed-protocol 3.0).

A dataset row stores one ROOT input seed per MLP. We expand it once into three
INDEPENDENT substreams so the estimator's randomness is decorrelated from both
the weight generation and the ground-truth sampling it is scored against:

    ss = SeedSequence(input_seed).spawn(3)
    ss[0] -> weight stream      (MLP weight matrices)
    ss[1] -> sample stream      (ground-truth Monte-Carlo draws)
    ss[2] -> estimator seed     (MLP.seed handed to the participant)

This module is the ONE place that computes ss[2]; bake and read both call it so
the derivation cannot drift.
"""

from __future__ import annotations

import flopscope.numpy as fnp


def derive_seed_streams(
    input_seed: int,
) -> "tuple[fnp.random.SeedSequence, fnp.random.SeedSequence, int]":
    """Return ``(weight_ss, sample_ss, estimator_seed)`` for a per-MLP input seed."""
    weight_ss, sample_ss, estimator_ss = fnp.random.SeedSequence(int(input_seed)).spawn(3)
    estimator_seed = int(estimator_ss.generate_state(1)[0])
    return weight_ss, sample_ss, estimator_seed


def derive_estimator_seed(input_seed: int) -> int:
    """The protocol-3.0 estimator seed (``MLP.seed``) for a per-MLP input seed."""
    return derive_seed_streams(input_seed)[2]
