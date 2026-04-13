"""Reference estimators for MLP mean prediction using mechestim.

This module provides tutorial estimator classes that predict per-layer
output means using analytical first-moment propagation through ReLU networks.

For a ReLU unit z = max(0, w^T x), if x ~ N(mu, Sigma):
    E[z] = mu_pre * Phi(mu_pre/sigma_pre) + sigma_pre * phi(mu_pre/sigma_pre)

where mu_pre = w^T mu, sigma_pre^2 = w^T Sigma w, Phi is the normal CDF,
and phi is the normal PDF.
"""

from __future__ import annotations

from typing import Optional

import mechestim as me

from .domain import MLP
from .sdk import BaseEstimator

# Abramowitz & Stegun approximation constants (formula 26.2.17)
_P = 0.2316419
_A1 = 0.319381530
_A2 = -0.356563782
_A3 = 1.781477937
_A4 = -1.821255978
_A5 = 1.330274429


def _norm_pdf(x: me.ndarray) -> me.ndarray:
    """Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2*pi)."""
    return me.exp(-0.5 * x * x) / me.sqrt(2.0 * me.pi)


def _norm_cdf(x: me.ndarray) -> me.ndarray:
    """Standard normal CDF using the Abramowitz & Stegun approximation.

    Uses only basic mechestim operations (exp, abs). Accurate to < 7.5e-8.
    """
    t = 1.0 / (1.0 + _P * me.abs(x))
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    pdf = me.exp(-0.5 * x * x) / me.sqrt(2.0 * me.pi)
    cdf = 1.0 - pdf * poly
    return me.where(x >= 0, cdf, 1.0 - cdf)


class MeanPropagationEstimator(BaseEstimator):
    """Mean propagation estimator for ReLU MLPs.

    Propagates means through each layer using the ReLU expectation formula
    with a diagonal variance approximation (assumes independent neurons).
    """

    def predict(self, mlp: MLP, budget: int) -> me.ndarray:
        """Predict per-layer output means via first-moment propagation through ReLU layers."""
        _ = budget
        width = mlp.width
        mu = me.zeros(width)
        var = me.ones(width)

        rows = []
        for w in mlp.weights:
            mu_pre = w.T @ mu
            var_pre = (w * w).T @ var
            var_pre = me.maximum(var_pre, 1e-12)
            sigma_pre = me.sqrt(var_pre)

            alpha = mu_pre / sigma_pre
            phi_alpha = _norm_pdf(alpha)
            Phi_alpha = _norm_cdf(alpha)

            # E[ReLU(z)]
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # Var[ReLU(z)]
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var = me.maximum(ez2 - mu * mu, 0.0)

            rows.append(mu)

        return me.stack(rows, axis=0)


class CovariancePropagationEstimator(BaseEstimator):
    """Full covariance propagation estimator for ReLU MLPs."""

    _COV_RESCALE_THRESHOLD = 1e100

    def predict(self, mlp: MLP, budget: int) -> me.ndarray:
        """Predict per-layer means via full covariance propagation through ReLU layers."""
        _ = budget
        width = mlp.width
        mu = me.zeros(width)
        cov = me.eye(width)
        log_scale = 0.0

        rows = []
        for w in mlp.weights:
            # Rescale before matmul to prevent overflow
            cov_diag = me.diag(cov)
            max_var_np = float(me.max(me.asarray(cov_diag)))
            if max_var_np > self._COV_RESCALE_THRESHOLD:
                s = float(me.sqrt(max_var_np))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(me.log(s))

            mu_pre = w.T @ mu
            cov_pre = w.T @ cov @ w
            var_pre = me.maximum(me.diag(cov_pre), 1e-12)
            sigma_pre = me.sqrt(var_pre)

            alpha = mu_pre / sigma_pre
            phi_alpha = _norm_pdf(alpha)
            Phi_alpha = _norm_cdf(alpha)

            # Post-ReLU means
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # Post-ReLU diagonal variance
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = me.maximum(ez2 - mu * mu, 0.0)

            # Approximate post-ReLU covariance
            sigma_np = me.asarray(sigma_pre, dtype=me.float64)
            Phi_np = me.asarray(Phi_alpha, dtype=me.float64)
            gain_np = me.where(sigma_np > 1e-12, Phi_np, 0.0)
            gain = me.array(gain_np.astype(me.float32))
            cov = me.multiply(me.outer(gain, gain), cov_pre)
            # Set diagonal to var_post (in-place, like numpy)
            me.fill_diagonal(cov, var_post)

            # Record mean in original (unscaled) coordinates
            scale_factor = float(me.exp(log_scale))
            rows.append(mu * scale_factor)

        return me.stack(rows, axis=0)


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

    def predict(self, mlp: MLP, budget: int) -> me.ndarray:
        """Route to covariance or mean propagation based on available FLOP budget."""
        if budget >= self._COVARIANCE_FLOP_MULTIPLIER * mlp.width * mlp.width:
            return self._covariance_estimator.predict(mlp, budget)
        return self._mean_estimator.predict(mlp, budget)
