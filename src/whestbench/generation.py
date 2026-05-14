"""Random MLP sampling utilities used by the evaluator."""

from __future__ import annotations

from typing import Optional

import flopscope.numpy as fnp

from .domain import MLP


def sample_mlp(
    width: int,
    depth: int,
    rng: Optional[fnp.random.Generator] = None,
    *,
    seed: int = 0,
) -> MLP:
    """Sample a random MLP with He-initialized weight matrices.

    Each weight matrix has shape ``(width, width)`` with entries drawn from
    ``N(0, 2/width)`` (He initialization for ReLU networks).

    Args:
        width: Neuron count per layer.
        depth: Number of weight matrices.
        rng: Optional flopscope RNG for weight sampling. If None, a fresh
            unseeded generator is used.
        seed: Per-MLP grader-supplied seed to attach to the returned MLP for
            estimator consumption. This does NOT control weight sampling
            (which uses ``rng``); it's a separate stream the estimator may
            consume for its own randomness.
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
    mlp = MLP(width=width, depth=depth, weights=weights, seed=int(seed))
    mlp.validate()
    return mlp
