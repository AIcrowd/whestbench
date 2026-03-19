"""Reference estimators for MLP mean prediction.

This module provides tutorial estimator classes that predict per-layer
output means using analytical moment propagation through ReLU networks.

- ``MeanPropagationEstimator``: first-moment propagation through ReLU.
- ``CovariancePropagationEstimator``: first + second moment propagation.
- ``CombinedEstimator``: budget-aware routing between the two.

For a ReLU unit z = max(0, w^T x), if x ~ N(mu, Sigma):
    E[z] = mu_pre * Phi(mu_pre/sigma_pre) + sigma_pre * phi(mu_pre/sigma_pre)

where mu_pre = w^T mu, sigma_pre^2 = w^T Sigma w, Phi is the normal CDF,
and phi is the normal PDF.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore[import-untyped]

from .domain import MLP
from .sdk import BaseEstimator


class MeanPropagationEstimator(BaseEstimator):
    """Mean propagation estimator for ReLU MLPs.

    Propagates means through each layer using the ReLU expectation formula
    with a diagonal variance approximation (assumes independent neurons).
    """

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        _ = budget
        width = mlp.width
        mu = np.zeros(width, dtype=np.float64)
        var = np.ones(width, dtype=np.float64)

        rows = []
        for w in mlp.weights:
            W = w.astype(np.float64)
            mu_pre = W.T @ mu
            var_pre = (W ** 2).T @ var
            var_pre = np.maximum(var_pre, 1e-12)
            sigma_pre = np.sqrt(var_pre)

            alpha = mu_pre / sigma_pre
            phi_alpha = norm.pdf(alpha)
            Phi_alpha = norm.cdf(alpha)

            # E[ReLU(z)]
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # Var[ReLU(z)]
            ez2 = (mu_pre ** 2 + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var = np.maximum(ez2 - mu ** 2, 0.0)

            rows.append(mu.astype(np.float32))

        return np.stack(rows, axis=0)


class CovariancePropagationEstimator(BaseEstimator):
    """Full covariance propagation estimator for ReLU MLPs."""

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        _ = budget
        width = mlp.width
        mu = np.zeros(width, dtype=np.float64)
        cov = np.eye(width, dtype=np.float64)

        rows = []
        for w in mlp.weights:
            W = w.astype(np.float64)
            mu_pre = W.T @ mu
            cov_pre = W.T @ cov @ W
            var_pre = np.maximum(np.diag(cov_pre), 1e-12)
            sigma_pre = np.sqrt(var_pre)

            alpha = mu_pre / sigma_pre
            phi_alpha = norm.pdf(alpha)
            Phi_alpha = norm.cdf(alpha)

            # Post-ReLU means
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # Post-ReLU diagonal variance
            ez2 = (mu_pre ** 2 + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = np.maximum(ez2 - mu ** 2, 0.0)

            # Approximate post-ReLU covariance
            gain = np.where(sigma_pre > 1e-12, Phi_alpha, 0.0)
            cov = np.outer(gain, gain) * cov_pre
            np.fill_diagonal(cov, var_post)

            rows.append(mu.astype(np.float32))

        return np.stack(rows, axis=0)


class CombinedEstimator(BaseEstimator):
    """Budget-aware hybrid estimator."""

    _COVARIANCE_BUDGET_MULTIPLIER = 30

    def __init__(
        self,
        *,
        mean_estimator: Optional[BaseEstimator] = None,
        covariance_estimator: Optional[BaseEstimator] = None,
    ) -> None:
        self._mean_estimator = mean_estimator or MeanPropagationEstimator()
        self._covariance_estimator = covariance_estimator or CovariancePropagationEstimator()

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        if budget >= self._COVARIANCE_BUDGET_MULTIPLIER * mlp.width:
            return self._covariance_estimator.predict(mlp, budget)
        return self._mean_estimator.predict(mlp, budget)
