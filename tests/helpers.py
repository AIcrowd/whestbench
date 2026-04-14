from __future__ import annotations

import whest as we

from whestbench.domain import MLP
from whestbench.generation import sample_mlp
from whestbench.simulation import run_mlp_all_layers


def make_mlp(width: int, depth: int, seed: int = 42) -> MLP:
    """Create a small MLP for testing with a fixed seed."""
    return sample_mlp(width=width, depth=depth, rng=we.random.default_rng(seed))


def exhaustive_means(mlp: MLP, n_samples: int = 10000) -> we.ndarray:
    """Compute empirical per-layer means via brute-force sampling.

    Returns shape ``(depth, width)``.
    """
    inputs = we.random.default_rng().standard_normal((n_samples, mlp.width)).astype(we.float32)
    layer_outputs = run_mlp_all_layers(mlp, inputs)
    return we.stack([out.mean(axis=0) for out in layer_outputs]).astype(we.float32)
