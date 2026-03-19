"""MLP execution helpers for batched forward passes and empirical moments.

These utilities run an MLP layer-by-layer over random inputs and expose
per-layer outputs/means used by score computation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP


def relu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Element-wise ReLU activation."""
    return np.maximum(x, np.float32(0.0))


def run_mlp(mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
    """Forward pass returning final-layer activations.

    Args:
        mlp: MLP to execute.
        inputs: Input matrix of shape ``(samples, mlp.width)``.

    Returns:
        Activations of shape ``(samples, mlp.width)`` after the last layer.
    """
    x = inputs
    for w in mlp.weights:
        x = relu(x @ w)
    return x


def run_mlp_all_layers(
    mlp: MLP, inputs: NDArray[np.float32]
) -> List[NDArray[np.float32]]:
    """Forward pass returning activations after each layer.

    Args:
        mlp: MLP to execute.
        inputs: Input matrix of shape ``(samples, mlp.width)``.

    Returns:
        List of ``depth`` arrays, each shape ``(samples, mlp.width)``.
    """
    x = inputs
    layers: List[NDArray[np.float32]] = []
    for w in mlp.weights:
        x = relu(x @ w)
        layers.append(x)
    return layers


def output_stats(
    mlp: MLP, n_samples: int
) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
    """Compute per-layer means and average variance of the final layer.

    Args:
        mlp: MLP to evaluate.
        n_samples: Number of random Gaussian N(0,1) input vectors.

    Returns:
        all_layer_means: shape ``(depth, width)`` — mean activations per layer.
        final_mean: shape ``(width,)`` — mean activations at the final layer.
        avg_variance: scalar — average per-neuron variance at the final layer,
            used for ``sampling_mse`` normalization.
    """
    inputs = np.random.randn(n_samples, mlp.width).astype(np.float32)
    layer_outputs = run_mlp_all_layers(mlp, inputs)
    all_layer_means = np.stack(
        [np.mean(out, axis=0) for out in layer_outputs]
    ).astype(np.float32)
    final_outputs = layer_outputs[-1]
    final_mean = np.mean(final_outputs, axis=0).astype(np.float32)
    avg_variance = float(np.mean(np.var(final_outputs, axis=0)))
    return all_layer_means, final_mean, avg_variance
