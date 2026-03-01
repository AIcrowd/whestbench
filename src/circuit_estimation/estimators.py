"""Class-based reference estimators for per-depth wire-mean prediction.

This module is intentionally tutorial-oriented. The three estimator classes
show a progression of ideas:

- ``MeanPropagationEstimator`` performs first-moment propagation (mean propagation).
- ``CovariancePropagationEstimator`` keeps means and pairwise covariance.
- ``CombinedEstimator`` switches between those methods based on budget.

Each estimator uses the streaming contract from ``BaseEstimator``:
``predict(circuit, budget)`` yields exactly one ``(width,)`` row per depth.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit, Layer
from .sdk import BaseEstimator


class MeanPropagationEstimator(BaseEstimator):
    """Mean propagation tutorial estimator.

    This is the simplest baseline and usually the best estimator to read first.
    We keep one value per wire, its expected value ``E[x]``, and update those
    means layer by layer. That gives a strong speed baseline with minimal
    machinery.

    For one output wire ``i``, the gate equation is:

        y_i = a_i * x_f + b_i * x_s + c_i + p_i * x_f * x_s

    Mean propagation uses the approximation:

        E[x_f * x_s] ~= E[x_f] * E[x_s]

    so the update is:

        m_i^(l+1) = E[y_i]
                  = a_i * m_f
                    + b_i * m_s
                    + c_i
                    + p_i * m_f * m_s

    where ``m_f = E[x_f]`` and ``m_s = E[x_s]``.

    Intuition: we model average behavior only. If dependencies between wires
    become strong, this approximation can drift because covariance is ignored.
    Runtime is ``O(depth * width)`` with ``O(width)`` rolling state.
    """

    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        """Yield one mean vector per depth.

        The ``budget`` argument is accepted for API compatibility but is not
        used by this estimator.
        """
        _ = budget
        # Start from unbiased wire means E[x] = 0 at depth 0.
        x_mean: NDArray[np.float32] = np.zeros(circuit.n, dtype=np.float32)
        for layer in circuit.gates:
            # Push means through this layer's affine+bivariate gate map.
            x_mean = self._propagate_layer_mean(layer, x_mean)
            yield x_mean

    @staticmethod
    def _propagate_layer_mean(layer: Layer, x_mean: NDArray[np.float32]) -> NDArray[np.float32]:
        """Propagate means through one layer under first-moment closure."""
        first_mean = np.take(x_mean, layer.first)
        second_mean = np.take(x_mean, layer.second)
        return (
            layer.first_coeff * first_mean
            + layer.second_coeff * second_mean
            + layer.const
            + layer.product_coeff * first_mean * second_mean
        ).astype(np.float32)


class CovariancePropagationEstimator(BaseEstimator):
    """Pairwise moment closure tutorial estimator (mean + covariance).

    This estimator is the natural upgrade from mean propagation. In addition to
    per-wire means, it tracks covariance between wires, which captures whether
    two wires tend to move together or in opposite directions.

    Intuition: product gates depend on ``x_f * x_s``. If ``x_f`` and ``x_s``
    are often aligned, the product tends positive; if anti-aligned, negative.
    Mean-only propagation cannot represent that effect, but covariance can.

    State per depth:

        m = E[x]      (shape n)
        C = Cov[x]    (shape n x n)

    Gate model:

        y_i = a_i * x_f + b_i * x_s + c_i + p_i * x_f * x_s

    Mean update:

        E[y_i] = a_i * m_f
                 + b_i * m_s
                 + c_i
                 + p_i * (m_f * m_s + C_fs)

    Covariance update is decomposed into three blocks:
    linear-linear, linear-bilinear (1v2), and bilinear-bilinear (2v2).
    The helper methods mirror those blocks so each code path maps to one part
    of the approximation.

    Runtime is ``O(depth * n^2)`` with ``O(n^2)`` rolling state. This is
    usually more accurate than mean propagation but more expensive.
    """

    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        """Yield one mean vector per depth while propagating covariance state."""
        _ = budget
        n = circuit.n
        # Initialize moment state for random {-1,+1} inputs.
        x_mean: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
        x_cov: NDArray[np.float32] = np.eye(n, dtype=np.float32)

        for layer in circuit.gates:
            x_mean, x_cov = self._propagate_layer(layer, x_mean, x_cov)
            yield x_mean

    def _propagate_layer(
        self,
        layer: Layer,
        x_mean: NDArray[np.float32],
        x_cov: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return ``(new_mean, new_cov)`` for one layer under pairwise closure."""
        n = x_mean.shape[0]
        first_mean = x_mean[layer.first]
        second_mean = x_mean[layer.second]
        pair_cov = x_cov[layer.first, layer.second]

        # Step 1: propagate means with covariance-corrected product expectation.
        new_mean: NDArray[np.float32] = (
            layer.first_coeff * first_mean
            + layer.second_coeff * second_mean
            + layer.const
            + layer.product_coeff * first_mean * second_mean
            + layer.product_coeff * pair_cov
        ).astype(np.float32)

        # Step 2: start covariance with linear-linear contributions.
        new_cov: NDArray[np.float32] = self._linear_linear_covariance(layer, x_cov, n)

        # Step 3: add linear-bilinear cross contributions (1v2 terms).
        result_1v2_first = np.outer(
            layer.first_coeff, layer.product_coeff
        ) * self._one_v_two_covariance(
            layer.first,
            layer.first,
            layer.second,
            x_cov,
            x_mean,
        )
        new_cov += result_1v2_first + result_1v2_first.T

        result_1v2_second = np.outer(
            layer.second_coeff,
            layer.product_coeff,
        ) * self._one_v_two_covariance(layer.second, layer.first, layer.second, x_cov, x_mean)
        new_cov += result_1v2_second + result_1v2_second.T

        # Step 4: add bilinear-bilinear contribution (2v2 term).
        new_cov += np.outer(layer.product_coeff, layer.product_coeff) * self._two_v_two_covariance(
            layer.first,
            layer.second,
            layer.first,
            layer.second,
            x_cov,
            x_mean,
        )

        # Step 5: clip moments back into a feasible/stable region.
        self._clip_moments(new_mean, new_cov)
        return new_mean, new_cov

    @staticmethod
    def _linear_linear_covariance(
        layer: Layer,
        x_cov: NDArray[np.float32],
        n: int,
    ) -> NDArray[np.float32]:
        """Compute covariance contribution from linear terms only."""
        new_cov = np.zeros((n, n), dtype=np.float32)
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
        return new_cov

    @staticmethod
    def _one_v_two_covariance(
        a: NDArray[np.int32],
        b: NDArray[np.int32],
        c: NDArray[np.int32],
        x_cov: NDArray[np.float32],
        x_mean: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Approximate ``Cov(x[a], x[b] * x[c])`` under pairwise closure."""
        return (
            x_mean[b][None, :] * x_cov[np.ix_(a, c)] + x_mean[c][None, :] * x_cov[np.ix_(a, b)]
        ).astype(np.float32)

    @staticmethod
    def _two_v_two_covariance(
        a: NDArray[np.int32],
        b: NDArray[np.int32],
        c: NDArray[np.int32],
        d: NDArray[np.int32],
        cov: NDArray[np.float32],
        mean: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Approximate ``Cov(x[a]x[b], x[c]x[d])`` under pairwise closure."""
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

    @staticmethod
    def _clip_moments(mean: NDArray[np.float32], cov: NDArray[np.float32]) -> None:
        """Project moments to feasible bounds for ``{-1, +1}`` wire values."""
        n = len(mean)
        np.clip(mean, -1.0, 1.0, out=mean)
        var = 1.0 - mean * mean
        cov[np.arange(n), np.arange(n)] = var
        std = np.sqrt(np.clip(var, 0.0, None))
        max_cov = np.outer(std, std)
        np.clip(cov, -max_cov, max_cov, out=cov)


class CombinedEstimator(BaseEstimator):
    """Budget-aware hybrid tutorial estimator.

    This estimator demonstrates a practical contest pattern: pick a fast method
    for small budgets and a richer method for larger budgets.

    Routing rule:

        if budget >= 30 * width: use covariance propagation
        else:                    use mean propagation

    Mean propagation (fast path) uses:

        m_i^(l+1) = a_i * m_f
                    + b_i * m_s
                    + c_i
                    + p_i * m_f * m_s

    Covariance propagation (accurate path) adds second-order state and uses:

        E[x_f * x_s] ~= m_f * m_s + C_fs

    plus decomposed covariance terms (linear-linear, 1v2, 2v2).

    Intuition: budget controls how much structure we can afford to model.
    Low budget favors speed, high budget can afford covariance for better
    accuracy.
    """

    _COVARIANCE_BUDGET_MULTIPLIER = 30

    def __init__(
        self,
        *,
        mean_estimator: BaseEstimator | None = None,
        covariance_estimator: BaseEstimator | None = None,
    ) -> None:
        self._mean_estimator = mean_estimator or MeanPropagationEstimator()
        self._covariance_estimator = covariance_estimator or CovariancePropagationEstimator()

    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        """Choose estimator path by budget and stream depth rows.

        Keeping the `if` condition inline makes the budget-regime policy
        explicit for participants reading this class.
        """
        if budget >= self._COVARIANCE_BUDGET_MULTIPLIER * circuit.n:
            yield from self._covariance_estimator.predict(circuit, budget)
            return
        yield from self._mean_estimator.predict(circuit, budget)
