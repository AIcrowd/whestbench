"""Reference estimators for per-layer wire-mean prediction.

This module provides two complementary approaches:

- ``mean_propagation``: first-moment propagation only (fast, coarse).
- ``covariance_propagation``: tracks means plus an approximate covariance matrix
  using pairwise moment closure (slower, usually more accurate).

``combined_estimator`` selects between them using the runtime budget.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit


def mean_propagation(circuit: Circuit) -> NDArray[np.float32]:
    """Run first-moment propagation through all layers.

    This approximation only tracks ``E[x]`` for each wire and substitutes
    ``E[x_i x_j] ≈ E[x_i] E[x_j]``. It is computationally cheap and often
    useful at low budget where covariance tracking is too expensive.
    """
    x_mean: NDArray[np.float32] = np.zeros(circuit.n, dtype=np.float32)
    outputs = np.zeros((circuit.d, circuit.n), dtype=np.float32)
    for i, layer in enumerate(circuit.gates):
        # Tutorial note: this is the direct layer equation with random wires
        # replaced by their current means.
        first_mean = np.take(x_mean, layer.first)
        second_mean = np.take(x_mean, layer.second)
        x_mean = (
            layer.first_coeff * first_mean
            + layer.second_coeff * second_mean
            + layer.const
            + layer.product_coeff * first_mean * second_mean
        ).astype(np.float32)
        outputs[i] = x_mean
    return outputs


def one_v_two_covariance(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
    c: NDArray[np.int32],
    x_cov: NDArray[np.float32],
    x_mean: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Approximate ``Cov(x[a], x[b] * x[c])`` under pairwise moment closure.

    Under a pairwise moment closure, third-order structure is decomposed into
    products of first and second moments. This helper is used by the covariance
    estimator when linear terms interact with bilinear terms.
    """
    return (
        x_mean[b][None, :] * x_cov[np.ix_(a, c)] + x_mean[c][None, :] * x_cov[np.ix_(a, b)]
    ).astype(np.float32)


def two_v_two_covariance(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
    c: NDArray[np.int32],
    d: NDArray[np.int32],
    cov: NDArray[np.float32],
    mean: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Approximate ``Cov(x[a]x[b], x[c]x[d])`` with pairwise factors.

    This is the fourth-order companion to ``one_v_two_covariance``. It expands
    product-product covariance using combinations of pairwise covariances and
    means, consistent with the same closure assumption.
    """
    cov_ac = cov[np.ix_(a, c)]
    cov_ad = cov[np.ix_(a, d)]
    cov_bc = cov[np.ix_(b, c)]
    cov_bd = cov[np.ix_(b, d)]

    mu_a = mean[a][:, None]
    mu_b = mean[b][:, None]
    mu_c = mean[c][None, :]
    mu_d = mean[d][None, :]

    return (
        (mu_a * mu_c) * cov_bd
        + (mu_a * mu_d) * cov_bc
        + (mu_b * mu_c) * cov_ad
        + (mu_b * mu_d) * cov_ac
    ).astype(np.float32)


def clip(mean: NDArray[np.float32], cov: NDArray[np.float32]) -> None:
    """Project moments to feasible bounds for ``{-1, +1}``-valued wires.

    For signed wires, each variance is ``1 - mean^2`` and covariance must stay
    within ``[-std_i * std_j, +std_i * std_j]``. This guard keeps numerical
    drift from producing impossible moments.
    """
    n = len(mean)
    np.clip(mean, -1.0, 1.0, out=mean)
    var = 1.0 - mean * mean
    cov[np.arange(n), np.arange(n)] = var
    std = np.sqrt(np.clip(var, 0.0, None))
    max_cov = np.outer(std, std)
    np.clip(cov, -max_cov, max_cov, out=cov)


def covariance_propagation(circuit: Circuit) -> NDArray[np.float32]:
    """Run mean+covariance propagation using pairwise closure approximations.

    Walkthrough:
    1. Maintain current mean vector ``x_mean`` and covariance matrix ``x_cov``.
    2. Compute ``new_mean`` from linear, constant, and bilinear contributions.
       The bilinear term includes both ``E[x_i]E[x_j]`` and ``Cov(x_i, x_j)``.
    3. Build ``new_cov`` from:
       - linear-linear interactions,
       - linear-bilinear interactions (1v2 terms),
       - bilinear-bilinear interactions (2v2 terms).
    4. Clip to feasible ranges, then iterate.
    """
    n = circuit.n
    x_mean: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
    x_cov: NDArray[np.float32] = np.eye(n, dtype=np.float32)
    outputs = np.zeros((circuit.d, n), dtype=np.float32)

    for i, layer in enumerate(circuit.gates):
        first_mean = x_mean[layer.first]
        second_mean = x_mean[layer.second]
        pair_cov = x_cov[layer.first, layer.second]

        # Mean update: linear terms + constant + product expectation.
        # E[xy] = E[x]E[y] + Cov(x, y), so the bilinear contribution has two parts.
        new_mean: NDArray[np.float32] = (
            layer.first_coeff * first_mean
            + layer.second_coeff * second_mean
            + layer.const
            + layer.product_coeff * first_mean * second_mean
            + layer.product_coeff * pair_cov
        ).astype(np.float32)

        new_cov: NDArray[np.float32] = np.zeros((n, n), dtype=np.float32)
        # Linear-linear covariance terms.
        new_cov += (
            np.outer(layer.first_coeff, layer.first_coeff) * x_cov[np.ix_(layer.first, layer.first)]
        )
        new_cov += (
            np.outer(layer.second_coeff, layer.second_coeff)
            * x_cov[np.ix_(layer.second, layer.second)]
        )
        new_cov += (
            np.outer(layer.first_coeff, layer.second_coeff)
            * x_cov[np.ix_(layer.first, layer.second)]
        )
        new_cov += (
            np.outer(layer.second_coeff, layer.first_coeff)
            * x_cov[np.ix_(layer.second, layer.first)]
        )

        # Linear-bilinear interactions under pairwise moment closure.
        result_1v2_first = np.outer(layer.first_coeff, layer.product_coeff) * one_v_two_covariance(
            layer.first, layer.first, layer.second, x_cov, x_mean
        )
        new_cov += result_1v2_first + result_1v2_first.T

        result_1v2_second = np.outer(
            layer.second_coeff, layer.product_coeff
        ) * one_v_two_covariance(layer.second, layer.first, layer.second, x_cov, x_mean)
        new_cov += result_1v2_second + result_1v2_second.T

        # Bilinear-bilinear interaction term.
        new_cov += np.outer(layer.product_coeff, layer.product_coeff) * two_v_two_covariance(
            layer.first, layer.second, layer.first, layer.second, x_cov, x_mean
        )

        # Keep moments feasible for signed wire variables before next step.
        clip(new_mean, new_cov)
        x_mean, x_cov = new_mean, new_cov
        outputs[i] = x_mean

    return outputs


def combined_estimator(circuit: Circuit, budget: int) -> NDArray[np.float32]:
    """Dispatch estimator mode by budget.

    Heuristic:
    - use first-moment propagation at small budget for speed,
    - switch to covariance propagation when budget is large enough to support
      the extra ``O(n^2)`` state updates.
    """
    if budget >= 30 * circuit.n:
        return covariance_propagation(circuit)
    return mean_propagation(circuit)
