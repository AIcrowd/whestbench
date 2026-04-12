from __future__ import annotations

import mechestim as me

from whestbench.domain import MLP
from whestbench.generation import sample_mlp
from whestbench.simulation import run_mlp_all_layers


def make_mlp(width: int, depth: int, seed: int = 42) -> MLP:
    """Create a small MLP for testing with a fixed seed."""
    return sample_mlp(width=width, depth=depth, rng=me.random.default_rng(seed))


def exhaustive_means(mlp: MLP, n_samples: int = 10000) -> me.ndarray:
    """Compute empirical per-layer means via brute-force sampling.

    Returns shape ``(depth, width)``.
    """
    inputs = me.random.default_rng().standard_normal((n_samples, mlp.width)).astype(me.float32)
    layer_outputs = run_mlp_all_layers(mlp, inputs)
    return me.stack([out.mean(axis=0) for out in layer_outputs]).astype(me.float32)
