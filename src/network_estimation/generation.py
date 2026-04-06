"""Random MLP sampling utilities used by the evaluator."""

from __future__ import annotations

from typing import Optional

import mechestim as me

from .domain import MLP


def sample_mlp(width: int, depth: int, rng: Optional[me.random.Generator] = None) -> MLP:
    """Sample a random MLP with He-initialized weight matrices.

    Each weight matrix has shape ``(width, width)`` with entries drawn from
    ``N(0, 2/width)`` (He initialization for ReLU networks).

    Uses np.random for seeded generation (reproducibility), then wraps
    in me.array(). Array creation is free (0 FLOPs) in mechestim.
    """
    if width <= 0:
        raise ValueError("width must be positive.")
    if depth <= 0:
        raise ValueError("depth must be positive.")
    rng = rng or me.random.default_rng()
    scale = float(me.sqrt(2.0 / width))
    weights = [
        me.array((rng.standard_normal((width, width)) * scale).astype(me.float32))
        for _ in range(depth)
    ]
    mlp = MLP(width=width, depth=depth, weights=weights)
    mlp.validate()
    return mlp
