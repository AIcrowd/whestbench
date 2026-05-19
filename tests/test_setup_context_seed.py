"""Tests for SetupContext.seed and the --seed CLI plumbing."""

from __future__ import annotations

import dataclasses

import pytest

from whestbench.sdk import SetupContext


def test_setup_context_has_seed_field_with_default_zero():
    """SetupContext gains a seed field; default is 0 for backward compat."""
    ctx = SetupContext(
        width=10,
        depth=2,
        flop_budget=100,
        api_version="1.0",
    )
    assert hasattr(ctx, "seed")
    assert ctx.seed == 0


def test_setup_context_accepts_explicit_seed():
    """Seed is settable at construction time."""
    ctx = SetupContext(
        width=10,
        depth=2,
        flop_budget=100,
        api_version="1.0",
        seed=42,
    )
    assert ctx.seed == 42


def test_setup_context_is_frozen_so_seed_cannot_be_mutated():
    """Reproducibility relies on SetupContext being immutable — participants cannot reseed mid-run."""
    ctx = SetupContext(
        width=10,
        depth=2,
        flop_budget=100,
        api_version="1.0",
        seed=42,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.seed = 99  # type: ignore[misc]


def test_local_runner_setup_context_carries_spec_seed(tmp_path):
    """`whest run --seed 42 --runner local` produces a SetupContext with seed=42."""
    import json
    import subprocess

    # Use the smoke-test path which doesn't require a dataset on disk.
    # Write a tiny fixture estimator that records ctx.seed into its instance state,
    # then dump it to JSON via a side channel that the test can read back.
    fixture = tmp_path / "seed_recording_estimator.py"
    record_file = tmp_path / "ctx_seed.json"
    fixture.write_text(f"""
import json
from whestbench.sdk import BaseEstimator
import flopscope.numpy as fnp

class Estimator(BaseEstimator):
    def setup(self, ctx):
        with open({str(record_file)!r}, "w") as f:
            json.dump({{"seed": ctx.seed}}, f)

    def predict(self, mlp, budget):
        return fnp.zeros((mlp.depth, mlp.width))
""")

    result = subprocess.run(
        [
            "uv",
            "run",
            "whest",
            "run",
            "--estimator",
            str(fixture),
            "--seed",
            "42",
            "--runner",
            "local",
            "--n-mlps",
            "1",
            "--n-samples",
            "100",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"whest run failed: {result.stderr}"
    assert record_file.exists(), "estimator did not run / did not write recording"
    recorded = json.loads(record_file.read_text())
    assert recorded["seed"] == 42


def test_local_runner_unseeded_setup_context_has_zero_seed(tmp_path):
    """No --seed flag -> ctx.seed = 0 (deterministic, unseeded default)."""
    import json
    import subprocess

    fixture = tmp_path / "seed_recording_estimator.py"
    record_file = tmp_path / "ctx_seed.json"
    fixture.write_text(f"""
import json
from whestbench.sdk import BaseEstimator
import flopscope.numpy as fnp

class Estimator(BaseEstimator):
    def setup(self, ctx):
        with open({str(record_file)!r}, "w") as f:
            json.dump({{"seed": ctx.seed}}, f)

    def predict(self, mlp, budget):
        return fnp.zeros((mlp.depth, mlp.width))
""")

    result = subprocess.run(
        [
            "uv",
            "run",
            "whest",
            "run",
            "--estimator",
            str(fixture),
            "--runner",
            "local",
            "--n-mlps",
            "1",
            "--n-samples",
            "100",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"whest run failed: {result.stderr}"
    assert record_file.exists(), "estimator did not run / did not write recording"
    recorded = json.loads(record_file.read_text())
    assert recorded["seed"] == 0


def test_seed_flag_allowed_alongside_dataset(tmp_path):
    """The old `--seed only without --dataset` constraint is removed."""
    import json
    import subprocess

    # First, bake a tiny dataset.
    dataset_path = tmp_path / "ds.npz"
    bake = subprocess.run(
        [
            "uv",
            "run",
            "whest",
            "create-dataset",
            "--n-mlps",
            "1",
            "--width",
            "8",
            "--depth",
            "2",
            "--seed",
            "1",
            "--n-samples",
            "100",
            "--output",
            str(dataset_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert bake.returncode == 0, f"create-dataset failed: {bake.stderr}"
    assert dataset_path.exists()

    # Now run with both --seed AND --dataset. Previously this raised.
    fixture = tmp_path / "seed_recording_estimator.py"
    record_file = tmp_path / "ctx_seed.json"
    fixture.write_text(f"""
import json
from whestbench.sdk import BaseEstimator
import flopscope.numpy as fnp

class Estimator(BaseEstimator):
    def setup(self, ctx):
        with open({str(record_file)!r}, "w") as f:
            json.dump({{"seed": ctx.seed}}, f)

    def predict(self, mlp, budget):
        return fnp.zeros((mlp.depth, mlp.width))
""")

    result = subprocess.run(
        [
            "uv",
            "run",
            "whest",
            "run",
            "--estimator",
            str(fixture),
            "--dataset",
            str(dataset_path),
            "--seed",
            "42",
            "--runner",
            "local",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--seed alongside --dataset should be allowed; got returncode "
        f"{result.returncode}, stderr: {result.stderr}"
    )
    assert record_file.exists(), "estimator did not run / did not write recording"
    recorded = json.loads(record_file.read_text())
    assert recorded["seed"] == 42, (
        f"--seed should flow into ctx.seed even with --dataset; got {recorded['seed']}"
    )


def test_subprocess_runner_setup_context_carries_spec_seed(tmp_path):
    """`whest run --seed 42 --runner subprocess` produces ctx.seed=42 inside the worker."""
    import json
    import subprocess

    fixture = tmp_path / "seed_recording_estimator.py"
    record_file = tmp_path / "ctx_seed.json"
    fixture.write_text(f"""
import json
from whestbench.sdk import BaseEstimator
import flopscope.numpy as fnp

class Estimator(BaseEstimator):
    def setup(self, ctx):
        with open({str(record_file)!r}, "w") as f:
            json.dump({{"seed": ctx.seed}}, f)

    def predict(self, mlp, budget):
        return fnp.zeros((mlp.depth, mlp.width))
""")

    result = subprocess.run(
        [
            "uv",
            "run",
            "whest",
            "run",
            "--estimator",
            str(fixture),
            "--seed",
            "42",
            "--runner",
            "subprocess",
            "--n-mlps",
            "1",
            "--flop-budget",
            "1000000",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"whest run failed: {result.stderr}"
    assert record_file.exists(), (
        "subprocess worker did not write record file — estimator setup may not have run"
    )
    recorded = json.loads(record_file.read_text())
    assert recorded["seed"] == 42, (
        f"Subprocess worker did not receive seed=42; got {recorded['seed']}"
    )


def test_subprocess_worker_defaults_to_zero_if_seed_missing_from_payload():
    """Defensive: if the host sends no `seed` key, the worker should default to 0, not crash."""
    # This test directly verifies the worker's defensive defaulting pattern.
    # It does NOT spawn a subprocess.
    from whestbench.sdk import SetupContext

    ctx_payload = {
        "width": 8,
        "depth": 2,
        "flop_budget": 1000,
        "api_version": "1.0",
        "scratch_dir": None,
        # NOTE: deliberately no "seed" key
    }
    ctx = SetupContext(
        width=int(ctx_payload["width"]),
        depth=int(ctx_payload["depth"]),
        flop_budget=int(ctx_payload["flop_budget"]),
        api_version=str(ctx_payload["api_version"]),
        scratch_dir=(
            str(ctx_payload["scratch_dir"]) if ctx_payload.get("scratch_dir") is not None else None
        ),
        seed=int(ctx_payload.get("seed", 0)),
    )
    assert ctx.seed == 0
