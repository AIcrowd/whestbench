"""Estimator implementations for per-layer circuit mean prediction."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit


def mean_propagation(circuit: Circuit) -> Iterator[NDArray[np.float32]]:
    """Propagate only first moments through each layer."""
    x_mean: NDArray[np.float32] = np.zeros(circuit.n, dtype=np.float32)
    for layer in circuit.gates:
        x_mean = (
            layer.first_coeff * np.take(x_mean, layer.first)
            + layer.second_coeff * np.take(x_mean, layer.second)
            + layer.const
            + layer.product_coeff * np.take(x_mean, layer.first) * np.take(x_mean, layer.second)
        )
        yield x_mean


def one_v_two_covariance(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
    c: NDArray[np.int32],
    x_cov: NDArray[np.float32],
    x_mean: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Approximate Cov(x[a[i]], x[b[j]]*x[c[j]]) under pairwise moment closure."""
    return x_mean[b][None, :] * x_cov[np.ix_(a, c)] + x_mean[c][None, :] * x_cov[np.ix_(a, b)]


def two_v_two_covariance(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
    c: NDArray[np.int32],
    d: NDArray[np.int32],
    cov: NDArray[np.float32],
    mean: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Approximate Cov(x[a]x[b], x[c]x[d]) using pairwise covariance factors."""
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
    )


def clip(mean: NDArray[np.float32], cov: NDArray[np.float32]) -> None:
    """Clip means/covariances to feasible correlation bounds for signed wires."""
    n = len(mean)
    np.clip(mean, -1.0, 1.0, out=mean)
    var = 1.0 - mean * mean
    cov[np.arange(n), np.arange(n)] = var
    std = np.sqrt(np.clip(var, 0.0, None))
    max_cov = np.outer(std, std)
    np.clip(cov, -max_cov, max_cov, out=cov)


def covariance_propagation(circuit: Circuit) -> Iterator[NDArray[np.float32]]:
    """Propagate means and covariance approximation through layers."""
    n = circuit.n
    x_mean: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
    x_cov: NDArray[np.float32] = np.eye(n, dtype=np.float32)
    for layer in circuit.gates:
        new_mean: NDArray[np.float32] = (
            layer.first_coeff * x_mean[layer.first]
            + layer.second_coeff * x_mean[layer.second]
            + layer.const
            + layer.product_coeff * x_mean[layer.first] * x_mean[layer.second]
            + layer.product_coeff * x_cov[layer.first, layer.second]
        )

        new_cov: NDArray[np.float32] = np.zeros((n, n), dtype=np.float32)
        new_cov += np.outer(layer.first_coeff, layer.first_coeff) * x_cov[np.ix_(layer.first, layer.first)]
        new_cov += np.outer(layer.second_coeff, layer.second_coeff) * x_cov[
            np.ix_(layer.second, layer.second)
        ]
        new_cov += np.outer(layer.first_coeff, layer.second_coeff) * x_cov[
            np.ix_(layer.first, layer.second)
        ]
        new_cov += np.outer(layer.second_coeff, layer.first_coeff) * x_cov[
            np.ix_(layer.second, layer.first)
        ]

        result_1v2_first = np.outer(layer.first_coeff, layer.product_coeff) * one_v_two_covariance(
            layer.first, layer.first, layer.second, x_cov, x_mean
        )
        new_cov += result_1v2_first + result_1v2_first.T

        result_1v2_second = np.outer(layer.second_coeff, layer.product_coeff) * one_v_two_covariance(
            layer.second, layer.first, layer.second, x_cov, x_mean
        )
        new_cov += result_1v2_second + result_1v2_second.T

        new_cov += np.outer(layer.product_coeff, layer.product_coeff) * two_v_two_covariance(
            layer.first, layer.second, layer.first, layer.second, x_cov, x_mean
        )

        clip(new_mean, new_cov)
        x_mean, x_cov = new_mean, new_cov
        yield x_mean


def combined_estimator(circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
    """Switch estimators by budget: covariance mode for larger budgets."""
    if budget >= 30 * circuit.n:
        yield from covariance_propagation(circuit)
    else:
        yield from mean_propagation(circuit)
