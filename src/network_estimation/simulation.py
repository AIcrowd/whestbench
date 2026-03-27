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


def run_mlp_all_layers(mlp: MLP, inputs: NDArray[np.float32]) -> List[NDArray[np.float32]]:
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


def sample_layer_statistics(
    mlp: MLP, n_samples: int
) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
    """Estimate per-layer activation statistics via Monte Carlo sampling.

    Feeds ``n_samples`` random Gaussian inputs through the MLP and computes
    empirical statistics of the activations at each layer.  This is the
    reference (pure-NumPy) implementation — accelerated backends provide
    equivalent methods with chunked streaming for lower memory usage.

    The returned values are used in two places:

    * **Scoring** (``scoring.py``): ``final_mean`` and ``avg_variance``
      normalise the ``sampling_mse`` metric so that networks with
      naturally high variance are not unfairly penalised.
    * **Dataset generation** (``dataset.py``): ``all_layer_means`` captures
      the ground-truth activation profile that estimators try to predict.

    Args:
        mlp: The MLP network to evaluate.
        n_samples: How many i.i.d. N(0, 1) input vectors to draw.  Larger
            values give more precise estimates at the cost of compute time.

    Returns:
        all_layer_means: ``(depth, width)`` float32 array — the mean
            activation of every neuron at every layer, averaged over all
            samples.
        final_mean: ``(width,)`` float32 array — the mean activation at
            the last layer (equivalent to ``all_layer_means[-1]``).
        avg_variance: Scalar — the mean per-neuron variance at the final
            layer, used as a normalisation baseline for ``sampling_mse``.
    """
    inputs = np.random.default_rng().standard_normal((n_samples, mlp.width), dtype=np.float32)
    layer_outputs = run_mlp_all_layers(mlp, inputs)
    all_layer_means = np.stack([np.mean(out, axis=0) for out in layer_outputs]).astype(np.float32)
    final_outputs = layer_outputs[-1]
    final_mean = np.mean(final_outputs, axis=0).astype(np.float32)
    avg_variance = float(np.mean(np.var(final_outputs, axis=0)))
    return all_layer_means, final_mean, avg_variance
