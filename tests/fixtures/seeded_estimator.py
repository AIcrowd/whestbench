"""Test-only estimator: records the first setup-time RNG draw to a side-channel file.

Use this in integration tests that need to assert determinism of seeded setup
across runs. The estimator path on disk is what gets passed to ``whest run
--estimator <path>``; the side-channel file path is read from the
``WHEST_TEST_RECORD_FILE`` env var so the harness can vary it per test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import flopscope.numpy as fnp

from whestbench.sdk import BaseEstimator, SetupContext


class Estimator(BaseEstimator):
    def setup(self, context: SetupContext) -> None:
        # Build a Generator the same way the docs recommend.
        rng = fnp.random.default_rng(context.seed)
        # First draw — anything deterministic given the seed will do.
        sample = rng.normal(size=(3,))
        record_path = os.environ.get("WHEST_TEST_RECORD_FILE")
        if record_path is None:
            return  # silently no-op outside test harness
        Path(record_path).write_text(json.dumps({"seed": context.seed, "sample": sample.tolist()}))

    def predict(self, mlp, budget: int):  # noqa: ARG002
        return fnp.zeros((mlp.depth, mlp.width))
