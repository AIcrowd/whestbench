"""Scoring loop and baseline timing for MLP estimation contests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .generation import sample_mlp
from .sdk import BaseEstimator
from .simulation_fast import output_stats, run_mlp


@dataclass
class ContestSpec:
    """Evaluator configuration for one scoring run."""

    width: int
    depth: int
    n_mlps: int
    estimator_budget: int
    ground_truth_budget: int

    def validate(self) -> None:
        """Validate contest specification parameters are all positive."""
        if self.width <= 0:
            raise ValueError("width must be positive.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.n_mlps <= 0:
            raise ValueError("n_mlps must be positive.")
        if self.estimator_budget <= 0:
            raise ValueError("estimator_budget must be positive.")
        if self.ground_truth_budget <= 0:
            raise ValueError("ground_truth_budget must be positive.")


default_spec = ContestSpec(
    width=256,
    depth=16,
    n_mlps=10,
    estimator_budget=256 * 256 * 4,
    ground_truth_budget=256 * 256 * 256,
)


@dataclass
class ContestData:
    """Precomputed contest data for scoring."""

    spec: ContestSpec
    mlps: List[MLP]
    all_layer_targets: List[NDArray[np.float32]]
    final_targets: List[NDArray[np.float32]]
    avg_variances: List[float]


def make_contest(spec: ContestSpec) -> ContestData:
    """Generate MLPs and compute ground truth for a contest run."""
    spec.validate()
    mlps: List[MLP] = []
    all_layer_targets: List[NDArray[np.float32]] = []
    final_targets: List[NDArray[np.float32]] = []
    avg_variances: List[float] = []

    for _ in range(spec.n_mlps):
        mlp = sample_mlp(spec.width, spec.depth)
        all_means, final_mean, avg_var = output_stats(mlp, spec.ground_truth_budget)
        mlps.append(mlp)
        all_layer_targets.append(all_means)
        final_targets.append(final_mean)
        avg_variances.append(avg_var)

    return ContestData(
        spec=spec,
        mlps=mlps,
        all_layer_targets=all_layer_targets,
        final_targets=final_targets,
        avg_variances=avg_variances,
    )


def baseline_time(mlp: MLP, n_samples: int) -> float:
    """Measure wall time for a single forward pass with ``n_samples`` inputs."""
    inputs = np.random.randn(n_samples, mlp.width).astype(np.float32)
    t0 = time.perf_counter()
    run_mlp(mlp, inputs)
    return time.perf_counter() - t0


def validate_predictions(
    predictions: NDArray[np.float32], *, depth: int, width: int
) -> NDArray[np.float32]:
    """Validate estimator prediction array shape and finiteness."""
    arr = np.asarray(predictions, dtype=np.float32)
    if arr.shape != (depth, width):
        raise ValueError(
            f"Predictions must have shape ({depth}, {width}), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Predictions must contain only finite values.")
    return arr


def evaluate_estimator(
    estimator: BaseEstimator,
    data: ContestData,
) -> Dict[str, Any]:
    """Score an estimator against precomputed contest data."""
    spec = data.spec
    per_mlp: List[Dict[str, Any]] = []
    primary_scores: List[float] = []
    secondary_scores: List[float] = []

    for i, mlp in enumerate(data.mlps):
        time_budget = baseline_time(mlp, spec.estimator_budget)
        time_budget = max(time_budget, 1e-9)

        t0 = time.perf_counter()
        try:
            raw_predictions = estimator.predict(mlp, spec.estimator_budget)
            predictions = validate_predictions(
                raw_predictions, depth=spec.depth, width=spec.width
            )
        except Exception as exc:
            predictions = np.zeros((spec.depth, spec.width), dtype=np.float32)
            per_mlp.append({"mlp_index": i, "error": str(exc)})
            time_spent = time_budget
        else:
            time_spent = time.perf_counter() - t0

        # Time check: over budget -> zeros
        if time_spent > time_budget:
            predictions = np.zeros((spec.depth, spec.width), dtype=np.float32)

        # Time credit: floor at 50%
        fraction_spent = max(time_spent / time_budget, 0.5)

        # Normalization
        avg_var = data.avg_variances[i]
        sampling_mse = avg_var / (spec.estimator_budget * fraction_spent)
        sampling_mse = max(sampling_mse, 1e-30)

        # Primary score: final layer
        final_pred = predictions[-1]
        final_target = data.final_targets[i]
        final_mse = float(np.mean((final_pred - final_target) ** 2))
        primary = final_mse / sampling_mse

        # Secondary score: all layers
        all_target = data.all_layer_targets[i]
        all_mse = float(np.mean((predictions - all_target) ** 2))
        secondary = all_mse / sampling_mse

        primary_scores.append(primary)
        secondary_scores.append(secondary)

        if not per_mlp or per_mlp[-1].get("mlp_index") != i:
            per_mlp.append({
                "mlp_index": i,
                "time_budget_s": time_budget,
                "time_spent_s": time_spent,
                "fraction_spent": fraction_spent,
                "final_mse": final_mse,
                "all_layer_mse": all_mse,
                "primary_score": primary,
                "secondary_score": secondary,
            })

    return {
        "primary_score": float(np.mean(primary_scores)),
        "secondary_score": float(np.mean(secondary_scores)),
        "per_mlp": per_mlp,
    }
