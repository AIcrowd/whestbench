"""Reference estimators for MLP mean prediction using flopscope.

This module provides tutorial estimator classes that predict per-layer
output means using analytical first-moment propagation through ReLU networks.

For a ReLU unit z = max(0, w^T x), if x ~ N(mu, Sigma):
    E[z] = mu_pre * Phi(mu_pre/sigma_pre) + sigma_pre * phi(mu_pre/sigma_pre)

where mu_pre = w^T mu, sigma_pre^2 = w^T Sigma w, Phi is the normal CDF,
and phi is the normal PDF.
"""

from __future__ import annotations

from typing import Optional

import flopscope as flops
import flopscope.numpy as fnp

from .domain import MLP
from .sdk import BaseEstimator


class MeanPropagationEstimator(BaseEstimator):
    """Mean propagation estimator for ReLU MLPs.

    Propagates means through each layer using the ReLU expectation formula
    with a diagonal variance approximation (assumes independent neurons).
    """

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict per-layer output means via first-moment propagation through ReLU layers."""
        _ = budget
        width = mlp.width
        mu = fnp.zeros(width)
        var = fnp.ones(width)

        rows = []
        for w in mlp.weights:
            mu_pre = w.T @ mu
            var_pre = (w * w).T @ var
            var_pre = fnp.maximum(var_pre, 1e-12)
            sigma_pre = fnp.sqrt(var_pre)

            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)

            # E[ReLU(z)]
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # Var[ReLU(z)]
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var = fnp.maximum(ez2 - mu * mu, 0.0)

            rows.append(mu)

        return fnp.stack(rows, axis=0)


class CovariancePropagationEstimator(BaseEstimator):
    """Full covariance propagation estimator for ReLU MLPs."""

    _COV_RESCALE_THRESHOLD = 1e100

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict per-layer means via full covariance propagation through ReLU layers."""
        _ = budget
        width = mlp.width
        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        log_scale = 0.0

        rows = []
        for w in mlp.weights:
            # Rescale before matmul to prevent overflow
            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(fnp.asarray(cov_diag)))
            if max_var_np > self._COV_RESCALE_THRESHOLD:
                s = float(fnp.sqrt(max_var_np))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(fnp.log(s))

            mu_pre = w.T @ mu
            cov_pre = w.T @ cov @ w
            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)

            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)

            # Post-ReLU means
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # Post-ReLU diagonal variance
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)

            # Approximate post-ReLU covariance
            sigma_np = fnp.asarray(sigma_pre, dtype=fnp.float64)
            Phi_np = fnp.asarray(Phi_alpha, dtype=fnp.float64)
            gain_np = fnp.where(sigma_np > 1e-12, Phi_np, 0.0)
            gain = fnp.array(gain_np.astype(fnp.float32))
            cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            # Set diagonal to var_post (in-place, like numpy)
            fnp.fill_diagonal(cov, var_post)

            # Record mean in original (unscaled) coordinates
            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor)

        return fnp.stack(rows, axis=0)


class CombinedEstimator(BaseEstimator):
    """Budget-aware hybrid estimator.

    Routes to covariance propagation when the FLOP budget is large
    enough relative to width^2, otherwise falls back to mean propagation.
    """

    _COVARIANCE_FLOP_MULTIPLIER = 30

    def __init__(
        self,
        *,
        mean_estimator: Optional[BaseEstimator] = None,
        covariance_estimator: Optional[BaseEstimator] = None,
    ) -> None:
        self._mean_estimator = mean_estimator or MeanPropagationEstimator()
        self._covariance_estimator = covariance_estimator or CovariancePropagationEstimator()

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Route to covariance or mean propagation based on available FLOP budget."""
        if budget >= self._COVARIANCE_FLOP_MULTIPLIER * mlp.width * mlp.width:
            return self._covariance_estimator.predict(mlp, budget)
        return self._mean_estimator.predict(mlp, budget)
