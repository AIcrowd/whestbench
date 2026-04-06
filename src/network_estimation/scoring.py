"""Scoring loop and FLOP budget enforcement for MLP estimation contests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import mechestim as me
import numpy as np

from .domain import MLP
from .generation import sample_mlp
from .sdk import BaseEstimator
from .simulation_backends import get_backend


@dataclass
class ContestSpec:
    """Evaluator configuration for one scoring run."""

    width: int
    depth: int
    n_mlps: int
    flop_budget: int
    ground_truth_samples: int
    setup_timeout_s: float = 5.0
    predict_timeout_s: float = 30.0
    memory_limit_mb: int = 4096

    def validate(self) -> None:
        """Validate that all contest specification fields are positive and consistent."""
        if self.width <= 0:
            raise ValueError("width must be positive.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.n_mlps <= 0:
            raise ValueError("n_mlps must be positive.")
        if self.flop_budget <= 0:
            raise ValueError("flop_budget must be positive.")
        if self.ground_truth_samples <= 0:
            raise ValueError("ground_truth_samples must be positive.")


@dataclass
class ContestData:
    """Precomputed contest data for scoring."""

    spec: ContestSpec
    mlps: List[MLP]
    all_layer_targets: List[np.ndarray]
    final_targets: List[np.ndarray]
    avg_variances: List[float]


def make_contest(spec: ContestSpec) -> ContestData:
    """Generate MLPs and compute ground truth for a contest run."""
    spec.validate()
    backend = get_backend()
    mlps: List[MLP] = []
    all_layer_targets: List[np.ndarray] = []
    final_targets: List[np.ndarray] = []
    avg_variances: List[float] = []

    for _ in range(spec.n_mlps):
        mlp = sample_mlp(spec.width, spec.depth)
        with me.BudgetContext(flop_budget=int(1e15)):
            all_means, final_mean, avg_var = backend.sample_layer_statistics(
                mlp, spec.ground_truth_samples
            )
        # Convert to numpy for ground truth storage
        all_layer_targets.append(np.asarray(all_means, dtype=np.float32))
        final_targets.append(np.asarray(final_mean, dtype=np.float32))
        mlps.append(mlp)
        avg_variances.append(avg_var)

    return ContestData(
        spec=spec,
        mlps=mlps,
        all_layer_targets=all_layer_targets,
        final_targets=final_targets,
        avg_variances=avg_variances,
    )


def validate_predictions(
    predictions: me.ndarray, *, depth: int, width: int
) -> me.ndarray:
    """Validate estimator prediction array shape and finiteness."""
    shape = tuple(predictions.shape) if hasattr(predictions, "shape") else ()
    if shape != (depth, width):
        raise ValueError(f"Predictions must have shape ({depth}, {width}), got {shape}.")
    pred_np = np.asarray(predictions, dtype=np.float32)
    if not np.all(np.isfinite(pred_np)):
        raise ValueError("Predictions must contain only finite values.")
    return predictions


def evaluate_estimator(
    estimator: BaseEstimator,
    data: ContestData,
    on_mlp_scored: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """Score an estimator against precomputed contest data.

    Each MLP prediction runs under a BudgetContext. If the budget is
    exhausted, predictions are zeroed. Score = pure MSE (lower is better).
    """
    spec = data.spec
    per_mlp: List[Dict[str, Any]] = []
    primary_scores: List[float] = []
    secondary_scores: List[float] = []

    for i, mlp in enumerate(data.mlps):
        flops_used = 0
        budget_exhausted = False

        try:
            with me.BudgetContext(flop_budget=spec.flop_budget) as budget:
                raw_predictions = estimator.predict(mlp, spec.flop_budget)
                predictions = validate_predictions(
                    raw_predictions, depth=spec.depth, width=spec.width
                )
                flops_used = budget.flops_used
        except me.BudgetExhaustedError:
            predictions = me.zeros((spec.depth, spec.width))
            budget_exhausted = True
            flops_used = spec.flop_budget
        except Exception as exc:
            predictions = me.zeros((spec.depth, spec.width))
            per_mlp.append({
                "mlp_index": i,
                "error": str(exc),
                "flops_used": 0,
                "budget_exhausted": False,
            })
            primary_scores.append(float("inf"))
            secondary_scores.append(float("inf"))
            if on_mlp_scored is not None:
                on_mlp_scored(i + 1)
            continue

        # Convert predictions to numpy for MSE computation
        pred_np = np.asarray(predictions, dtype=np.float32)

        # Primary score: final layer MSE
        final_pred = pred_np[-1]
        final_target = data.final_targets[i]
        final_mse = float(np.mean((final_pred - final_target) ** 2))

        # Secondary score: all layers MSE
        all_target = data.all_layer_targets[i]
        all_mse = float(np.mean((pred_np - all_target) ** 2))

        primary_scores.append(final_mse)
        secondary_scores.append(all_mse)

        per_mlp.append({
            "mlp_index": i,
            "final_mse": final_mse,
            "all_layer_mse": all_mse,
            "flops_used": flops_used,
            "budget_exhausted": budget_exhausted,
        })

        if on_mlp_scored is not None:
            on_mlp_scored(i + 1)

    return {
        "primary_score": float(np.mean(primary_scores)) if primary_scores else float("inf"),
        "secondary_score": float(np.mean(secondary_scores)) if secondary_scores else float("inf"),
        "per_mlp": per_mlp,
    }
