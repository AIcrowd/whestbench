"""Backward-compatible estimator surface with monkeypatch-friendly wrappers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from circuit import Circuit
from circuit_estimation import estimators as _impl

clip = _impl.clip
mean_propagation = _impl.mean_propagation
covariance_propagation = _impl.covariance_propagation
one_v_two_covariance = _impl.one_v_two_covariance
two_v_two_covariance = _impl.two_v_two_covariance


def combined_estimator(circuit: Circuit, budget: int) -> NDArray[np.float32]:
    """Compatibility wrapper preserving monkeypatch behavior in legacy tests."""
    if budget >= 30 * circuit.n:
        return covariance_propagation(circuit)
    return mean_propagation(circuit)


__all__ = [
    "clip",
    "combined_estimator",
    "covariance_propagation",
    "mean_propagation",
    "one_v_two_covariance",
    "two_v_two_covariance",
]
