from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_estimation.domain import MLP
from network_estimation.generation import sample_mlp
from network_estimation.simulation import run_mlp_all_layers


def make_mlp(width: int, depth: int, seed: int = 42) -> MLP:
    """Create a small MLP for testing with a fixed seed."""
    return sample_mlp(width=width, depth=depth, rng=np.random.default_rng(seed))


def exhaustive_means(mlp: MLP, n_samples: int = 10000) -> NDArray[np.float32]:
    """Compute empirical per-layer means via brute-force sampling.

    Returns shape ``(depth, width)``.
    """
    inputs = np.random.randn(n_samples, mlp.width).astype(np.float32)
    layer_outputs = run_mlp_all_layers(mlp, inputs)
    return np.stack([out.mean(axis=0) for out in layer_outputs]).astype(np.float32)
