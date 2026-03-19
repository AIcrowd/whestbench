"""Random MLP sampling utilities used by the evaluator.

This module produces synthetic MLPs with He-initialized weight matrices
for ReLU activation networks. The same sampling path is used by baseline
timing and score evaluation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .domain import MLP


def sample_mlp(
    width: int, depth: int, rng: Optional[np.random.Generator] = None
) -> MLP:
    """Sample a random MLP with He-initialized weight matrices.

    Each weight matrix has shape ``(width, width)`` with entries drawn from
    ``N(0, 2/width)`` (He initialization for ReLU networks).

    Args:
        width: Number of neurons per layer.
        depth: Number of weight matrices (layers).
        rng: Optional NumPy generator for reproducible sampling.

    Returns:
        A validated ``MLP`` instance.
    """
    if width <= 0:
        raise ValueError("width must be positive.")
    if depth <= 0:
        raise ValueError("depth must be positive.")
    rng = rng or np.random.default_rng()
    scale = np.sqrt(2.0 / width)
    weights = [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]
    mlp = MLP(width=width, depth=depth, weights=weights)
    mlp.validate()
    return mlp
