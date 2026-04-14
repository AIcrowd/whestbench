"""Random MLP sampling utilities used by the evaluator."""

from __future__ import annotations

from typing import Optional

import whest as we

from .domain import MLP


def sample_mlp(width: int, depth: int, rng: Optional[we.random.Generator] = None) -> MLP:
    """Sample a random MLP with He-initialized weight matrices.

    Each weight matrix has shape ``(width, width)`` with entries drawn from
    ``N(0, 2/width)`` (He initialization for ReLU networks).

    Uses np.random for seeded generation (reproducibility), then wraps
    in we.array(). Array creation is free (0 FLOPs) in whest.
    """
    if width <= 0:
        raise ValueError("width must be positive.")
    if depth <= 0:
        raise ValueError("depth must be positive.")
    rng = rng or we.random.default_rng()
    scale = float(we.sqrt(2.0 / width))
    weights = [
        we.array((rng.standard_normal((width, width)) * scale).astype(we.float32))
        for _ in range(depth)
    ]
    mlp = MLP(width=width, depth=depth, weights=weights)
    mlp.validate()
    return mlp
