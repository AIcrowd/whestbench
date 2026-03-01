"""Execution and empirical-statistics helpers for circuits."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit


def run_batched(circuit: Circuit, inputs: NDArray[np.float16]) -> Iterator[NDArray[np.float16]]:
    """Yield batched layer outputs for the provided input batch."""
    x: NDArray[np.float16] = inputs
    for layer in circuit.gates:
        x = (
            layer.const
            + layer.first_coeff * x[:, layer.first]
            + layer.second_coeff * x[:, layer.second]
            + layer.product_coeff * x[:, layer.first] * x[:, layer.second]
        ).astype(np.float16)
        yield x


def run_on_random(circuit: Circuit, trials: int) -> Iterator[NDArray[np.float16]]:
    """Sample random {-1, +1} inputs and yield outputs per layer."""
    if trials <= 0:
        raise ValueError("trials must be positive.")
    inputs = np.random.choice([-1.0, 1.0], size=(trials, circuit.n)).astype(np.float16)
    yield from run_batched(circuit, inputs)


def empirical_mean(circuit: Circuit, trials: int) -> Iterator[NDArray[np.float32]]:
    """Return empirical per-wire mean for each layer under random inputs."""
    for output in run_on_random(circuit, trials):
        yield np.mean(output.astype(np.float32), axis=0)
