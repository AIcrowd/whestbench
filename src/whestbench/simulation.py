"""MLP execution helpers for batched forward passes and empirical moments.

These utilities run an MLP layer-by-layer over random inputs and expose
per-layer outputs/means used by score computation.
"""

from __future__ import annotations

from typing import List, Tuple

import mechestim as me

from .domain import MLP


def relu(x: me.ndarray) -> me.ndarray:
    """Element-wise ReLU activation."""
    return me.maximum(x, me.float32(0.0))


def run_mlp(mlp: MLP, inputs: me.ndarray) -> me.ndarray:
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


def run_mlp_all_layers(mlp: MLP, inputs: me.ndarray) -> List[me.ndarray]:
    """Forward pass returning activations after each layer.

    Args:
        mlp: MLP to execute.
        inputs: Input matrix of shape ``(samples, mlp.width)``.

    Returns:
        List of ``depth`` arrays, each shape ``(samples, mlp.width)``.
    """
    x = inputs
    layers: List[me.ndarray] = []
    for w in mlp.weights:
        x = relu(x @ w)
        layers.append(x)
    return layers


def _pick_chunk_size(width: int) -> int:
    """Choose chunk size to keep peak memory bounded (~4 MB per chunk)."""
    return max(1024, min(16384, 2**20 // width))


def sample_layer_statistics(mlp: MLP, n_samples: int) -> Tuple[me.ndarray, me.ndarray, float]:
    """Estimate per-layer activation statistics via chunked Monte Carlo sampling.

    Feeds ``n_samples`` random Gaussian inputs through the MLP in memory-bounded
    chunks and computes empirical statistics of the activations at each layer.

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
    width = mlp.width
    depth = mlp.depth
    chunk_size = _pick_chunk_size(width)

    layer_sums = me.zeros((depth, width), dtype=me.float64)
    final_sum_sq = me.zeros(width, dtype=me.float64)
    n_processed = 0

    for start in range(0, n_samples, chunk_size):
        n = min(chunk_size, n_samples - start)
        x = me.array(me.random.default_rng().standard_normal((n, width)).astype(me.float32))
        for layer_idx, w in enumerate(mlp.weights):
            x = me.maximum(me.matmul(x, w), 0.0)
            x_f64 = me.asarray(x, dtype=me.float64)
            layer_sums[layer_idx] += me.sum(x_f64, axis=0)
        x_f64 = me.asarray(x, dtype=me.float64)
        final_sum_sq += me.sum(x_f64**2, axis=0)
        n_processed += n

    layer_means = me.asarray(layer_sums / n_processed, dtype=me.float32)
    final_mean = layer_means[-1].copy()
    avg_variance = float(
        me.mean(final_sum_sq / n_processed - me.asarray(final_mean, dtype=me.float64) ** 2)
    )
    return layer_means, final_mean, avg_variance
