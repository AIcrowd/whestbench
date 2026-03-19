"""Circuit execution helpers for batched simulation and empirical moments.

These utilities execute a sampled circuit layer-by-layer over many random
inputs and expose per-layer outputs/means used by score computation.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit


def run_batched(circuit: Circuit, inputs: NDArray[np.float16]) -> Iterator[NDArray[np.float16]]:
    """Yield batched outputs after each layer for a fixed input matrix.

    Args:
        circuit: Circuit to execute.
        inputs: Input matrix of shape ``(trials, circuit.n)``.

    Yields:
        ``np.float16`` arrays of shape ``(trials, circuit.n)``, one per layer.
    """
    x: NDArray[np.float16] = inputs
    for layer in circuit.gates:
        # Runtime baseline and evaluator paths both use float16 execution to
        # keep simulation cost aligned with intended contest conditions.
        x = (
            layer.const
            + layer.first_coeff * x[:, layer.first]
            + layer.second_coeff * x[:, layer.second]
            + layer.product_coeff * x[:, layer.first] * x[:, layer.second]
        ).astype(np.float16)
        yield x


def run_on_random(circuit: Circuit, trials: int) -> Iterator[NDArray[np.float16]]:
    """Sample random ``{-1, +1}`` inputs and yield outputs after each layer.

    Args:
        circuit: Circuit to execute.
        trials: Number of random input vectors to draw.
    """
    if trials <= 0:
        raise ValueError("trials must be positive.")
    inputs = np.random.choice([-1.0, 1.0], size=(trials, circuit.n)).astype(np.float16)
    yield from run_batched(circuit, inputs)


def empirical_mean(circuit: Circuit, trials: int) -> Iterator[NDArray[np.float32]]:
    """Yield empirical per-wire means for each layer under random inputs.

    Means are accumulated from float16 simulation outputs but converted to
    float32 before reduction to avoid unnecessary precision loss in averaging.
    """
    for output in run_on_random(circuit, trials):
        yield np.mean(output.astype(np.float32), axis=0)
