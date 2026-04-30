"""Random MLP sampling utilities used by the evaluator."""

from __future__ import annotations

from typing import Optional

import flopscope.numpy as fnp

from .domain import MLP


def sample_mlp(width: int, depth: int, rng: Optional[fnp.random.Generator] = None) -> MLP:
    """Sample a random MLP with He-initialized weight matrices.

    Each weight matrix has shape ``(width, width)`` with entries drawn from
    ``N(0, 2/width)`` (He initialization for ReLU networks).

    Uses np.random for seeded generation (reproducibility), then wraps
    in fnp.array(). Array creation is free (0 FLOPs) in flopscope.
    """
    if width <= 0:
        raise ValueError("width must be positive.")
    if depth <= 0:
        raise ValueError("depth must be positive.")
    if rng is None:
        rng = fnp.random.default_rng()
    assert rng is not None  # narrows for pyright; flopscope's default_rng is untyped
    scale = float(fnp.sqrt(2.0 / width))
    weights = [
        fnp.array((rng.standard_normal((width, width)) * scale).astype(fnp.float32))
        for _ in range(depth)
    ]
    mlp = MLP(width=width, depth=depth, weights=weights)
    mlp.validate()
    return mlp
