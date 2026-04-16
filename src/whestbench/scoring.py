"""Scoring loop and FLOP budget enforcement for MLP estimation contests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import whest as we

from .domain import MLP
from .generation import sample_mlp
from .sdk import BaseEstimator
from .simulation import sample_layer_statistics

if TYPE_CHECKING:
    from .dataset import DatasetBundle


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
    wall_time_limit_s: Optional[float] = None
    untracked_time_limit_s: Optional[float] = None

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
        if self.wall_time_limit_s is not None and self.wall_time_limit_s <= 0:
            raise ValueError("wall_time_limit_s must be positive when provided.")
        if self.untracked_time_limit_s is not None and self.untracked_time_limit_s <= 0:
            raise ValueError("untracked_time_limit_s must be positive when provided.")


@dataclass
class ContestData:
    """Precomputed contest data for scoring."""

    spec: ContestSpec
    mlps: List[MLP]
    all_layer_targets: List[we.ndarray]
    final_targets: List[we.ndarray]
    avg_variances: List[float]
    sampling_budget_breakdown: Optional[Dict[str, Any]] = None


def make_contest(
    spec: ContestSpec,
    on_mlp_done: Optional[Callable[[int], None]] = None,
) -> ContestData:
    """Generate MLPs, compute ground truth, and collect sampling attribution."""
    spec.validate()
    mlps: List[MLP] = []
    all_layer_targets: List[we.ndarray] = []
    final_targets: List[we.ndarray] = []
    avg_variances: List[float] = []
    sampling_breakdowns: List[Dict[str, Any]] = []

    for i in range(spec.n_mlps):
        mlp = sample_mlp(spec.width, spec.depth)
        with we.BudgetContext(flop_budget=int(1e15)) as sampling_budget:
            with we.namespace("sampling"):
                with we.namespace("sample_layer_statistics"):
                    all_means, final_mean, avg_var = sample_layer_statistics(
                        mlp, spec.ground_truth_samples
                    )
        normalized_sampling = _normalize_sampling_budget_breakdown(
            sampling_budget.summary_dict(by_namespace=True)
        )
        if normalized_sampling is not None:
            sampling_breakdowns.append(normalized_sampling)
        # Convert to whest arrays for ground truth storage
        all_layer_targets.append(we.asarray(all_means, dtype=we.float32))
        final_targets.append(we.asarray(final_mean, dtype=we.float32))
        mlps.append(mlp)
        avg_variances.append(avg_var)
        if on_mlp_done is not None:
            on_mlp_done(i + 1)

    return ContestData(
        spec=spec,
        mlps=mlps,
        all_layer_targets=all_layer_targets,
        final_targets=final_targets,
        avg_variances=avg_variances,
        sampling_budget_breakdown=_aggregate_budget_breakdowns(sampling_breakdowns),
    )


def make_contest_from_bundle(
    spec: ContestSpec,
    bundle: "DatasetBundle",
    n_mlps: int,
) -> ContestData:
    """Build ContestData from a precomputed dataset bundle.

    Takes the first ``n_mlps`` entries from the bundle's MLPs and targets.
    Ground truth is not recomputed; ``sampling_budget_breakdown`` is ``None``.

    Raises ``ValueError`` if ``n_mlps`` is not in ``[1, bundle.n_mlps]`` or if
    ``spec`` is inconsistent with the bundle's width/depth.
    """
    if n_mlps <= 0:
        raise ValueError("n_mlps must be positive.")
    if n_mlps > bundle.n_mlps:
        raise ValueError(
            f"n_mlps={n_mlps} exceeds bundle size {bundle.n_mlps}; clamp before calling."
        )
    spec.validate()
    if spec.n_mlps != n_mlps:
        raise ValueError(f"spec.n_mlps ({spec.n_mlps}) must equal n_mlps ({n_mlps}).")

    mlps = list(bundle.mlps[:n_mlps])
    all_layer_targets = [
        we.asarray(bundle.all_layer_means[i], dtype=we.float32) for i in range(n_mlps)
    ]
    final_targets = [we.asarray(bundle.final_means[i], dtype=we.float32) for i in range(n_mlps)]
    avg_variances = list(bundle.avg_variances[:n_mlps])

    return ContestData(
        spec=spec,
        mlps=mlps,
        all_layer_targets=all_layer_targets,
        final_targets=final_targets,
        avg_variances=avg_variances,
        sampling_budget_breakdown=None,
    )


def validate_predictions(predictions: we.ndarray, *, depth: int, width: int) -> we.ndarray:
    """Validate estimator prediction array shape and finiteness."""
    shape = tuple(predictions.shape) if hasattr(predictions, "shape") else ()
    if shape != (depth, width):
        raise ValueError(f"Predictions must have shape ({depth}, {width}), got {shape}.")
    pred_np = we.asarray(predictions, dtype=we.float32)
    if not we.all(we.isfinite(pred_np)):
        raise ValueError("Predictions must contain only finite values.")
    return predictions


def _predict_stats_to_dict(stats: Any) -> Optional[Dict[str, Any]]:
    if stats is None:
        return None
    if isinstance(stats, dict):
        return stats
    extracted: Dict[str, Any] = {}
    for field in (
        "flops_used",
        "wall_time_s",
        "tracked_time_s",
        "untracked_time_s",
        "budget_breakdown",
    ):
        if hasattr(stats, field):
            extracted[field] = getattr(stats, field)
    return extracted or None


def _normalize_estimator_namespace(namespace: object) -> str:
    namespace_str = "" if namespace is None else str(namespace).strip()
    if namespace_str in {"", "null", "None"}:
        return "estimator.estimator-client"
    if namespace_str.startswith("estimator."):
        return namespace_str
    return f"estimator.{namespace_str}"


def _normalize_sampling_namespace(namespace: object) -> str:
    namespace_str = "" if namespace is None else str(namespace).strip()
    if namespace_str in {"", "null", "None"}:
        return "sampling.sample_layer_statistics"
    if namespace_str.startswith("sampling."):
        return namespace_str
    return f"sampling.{namespace_str}"


def _normalize_sampling_budget_breakdown(
    raw: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    normalized: Dict[str, Any] = {
        "flop_budget": int(raw.get("flop_budget", 0)),
        "flops_used": int(raw.get("flops_used", 0)),
        "flops_remaining": int(raw.get("flops_remaining", 0)),
        "wall_time_s": float(raw.get("wall_time_s", 0.0) or 0.0),
        "tracked_time_s": float(raw.get("tracked_time_s", 0.0) or 0.0),
        "untracked_time_s": float(raw.get("untracked_time_s", 0.0) or 0.0),
        "by_namespace": {},
    }
    by_namespace = raw.get("by_namespace") or {}
    for namespace, bucket in by_namespace.items():
        normalized_namespace = _normalize_sampling_namespace(namespace)
        merged_bucket = normalized["by_namespace"].setdefault(
            normalized_namespace,
            {"flops_used": 0, "calls": 0, "tracked_time_s": 0.0, "operations": {}},
        )
        merged_bucket["flops_used"] += int(bucket.get("flops_used", 0))
        merged_bucket["calls"] += int(bucket.get("calls", 0))
        merged_bucket["tracked_time_s"] += float(bucket.get("tracked_time_s", 0.0) or 0.0)
        for op_name, op_info in (bucket.get("operations") or {}).items():
            merged_op = merged_bucket["operations"].setdefault(
                op_name, {"flop_cost": 0, "calls": 0, "duration": 0.0}
            )
            merged_op["flop_cost"] += int(op_info.get("flop_cost", 0))
            merged_op["calls"] += int(op_info.get("calls", 0))
            merged_op["duration"] += float(op_info.get("duration", 0.0) or 0.0)
    if not normalized["by_namespace"] and normalized["flops_used"] > 0:
        normalized["by_namespace"]["sampling.sample_layer_statistics"] = {
            "flops_used": normalized["flops_used"],
            "calls": int(raw.get("calls", 0)),
            "tracked_time_s": normalized["tracked_time_s"],
            "operations": dict(raw.get("operations", {})),
        }
    return normalized


def _normalize_estimator_budget_breakdown(
    raw: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    normalized: Dict[str, Any] = {
        "flop_budget": int(raw.get("flop_budget", 0)),
        "flops_used": int(raw.get("flops_used", 0)),
        "flops_remaining": int(raw.get("flops_remaining", 0)),
        "wall_time_s": float(raw.get("wall_time_s", 0.0) or 0.0),
        "tracked_time_s": float(raw.get("tracked_time_s", 0.0) or 0.0),
        "untracked_time_s": float(raw.get("untracked_time_s", 0.0) or 0.0),
        "by_namespace": {},
    }
    by_namespace = raw.get("by_namespace") or {}
    for namespace, bucket in by_namespace.items():
        normalized_namespace = _normalize_estimator_namespace(namespace)
        merged_bucket = normalized["by_namespace"].setdefault(
            normalized_namespace,
            {"flops_used": 0, "calls": 0, "tracked_time_s": 0.0, "operations": {}},
        )
        merged_bucket["flops_used"] += int(bucket.get("flops_used", 0))
        merged_bucket["calls"] += int(bucket.get("calls", 0))
        merged_bucket["tracked_time_s"] += float(bucket.get("tracked_time_s", 0.0) or 0.0)
        for op_name, op_info in (bucket.get("operations") or {}).items():
            merged_op = merged_bucket["operations"].setdefault(
                op_name, {"flop_cost": 0, "calls": 0, "duration": 0.0}
            )
            merged_op["flop_cost"] += int(op_info.get("flop_cost", 0))
            merged_op["calls"] += int(op_info.get("calls", 0))
            merged_op["duration"] += float(op_info.get("duration", 0.0) or 0.0)
    if not normalized["by_namespace"] and normalized["flops_used"] > 0:
        normalized["by_namespace"]["estimator.estimator-client"] = {
            "flops_used": normalized["flops_used"],
            "calls": int(raw.get("calls", 0)),
            "tracked_time_s": normalized["tracked_time_s"],
            "operations": dict(raw.get("operations", {})),
        }
    return normalized


def _aggregate_budget_breakdowns(
    breakdowns: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not breakdowns:
        return None
    aggregate: Dict[str, Any] = {
        "flop_budget": 0,
        "flops_used": 0,
        "flops_remaining": 0,
        "wall_time_s": 0.0,
        "tracked_time_s": 0.0,
        "untracked_time_s": 0.0,
        "by_namespace": {},
    }
    for breakdown in breakdowns:
        aggregate["flop_budget"] += int(breakdown.get("flop_budget", 0))
        aggregate["flops_used"] += int(breakdown.get("flops_used", 0))
        aggregate["flops_remaining"] += int(breakdown.get("flops_remaining", 0))
        aggregate["wall_time_s"] += float(breakdown.get("wall_time_s", 0.0) or 0.0)
        aggregate["tracked_time_s"] += float(breakdown.get("tracked_time_s", 0.0) or 0.0)
        aggregate["untracked_time_s"] += float(breakdown.get("untracked_time_s", 0.0) or 0.0)
        for namespace, bucket in (breakdown.get("by_namespace") or {}).items():
            merged = aggregate["by_namespace"].setdefault(
                namespace,
                {"flops_used": 0, "calls": 0, "tracked_time_s": 0.0, "operations": {}},
            )
            merged["flops_used"] += int(bucket.get("flops_used", 0))
            merged["calls"] += int(bucket.get("calls", 0))
            merged["tracked_time_s"] += float(bucket.get("tracked_time_s", 0.0) or 0.0)
            for op_name, op_info in (bucket.get("operations") or {}).items():
                op_bucket = merged["operations"].setdefault(
                    op_name, {"flop_cost": 0, "calls": 0, "duration": 0.0}
                )
                op_bucket["flop_cost"] += int(op_info.get("flop_cost", 0))
                op_bucket["calls"] += int(op_info.get("calls", 0))
                op_bucket["duration"] += float(op_info.get("duration", 0.0) or 0.0)
    return aggregate


def evaluate_estimator(
    estimator: BaseEstimator,
    data: ContestData,
    on_mlp_scored: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """Score an estimator against precomputed contest data.

    Each MLP prediction runs under a BudgetContext. If the FLOP budget,
    wall-time limit, or untracked-time limit is exceeded, predictions are
    zeroed and the violation is recorded. Score = pure MSE (lower is better).
    """
    spec = data.spec
    per_mlp: List[Dict[str, Any]] = []
    primary_scores: List[float] = []
    secondary_scores: List[float] = []
    normalized_breakdowns: List[Dict[str, Any]] = []
    last_predict_stats = getattr(estimator, "last_predict_stats", None)

    for i, mlp in enumerate(data.mlps):
        flops_used = 0
        budget_exhausted = False
        time_exhausted = False
        untracked_time_exhausted = False
        raw_breakdown: Optional[Dict[str, Any]] = None
        normalized_breakdown: Optional[Dict[str, Any]] = None

        budget_ctx = we.BudgetContext(
            flop_budget=spec.flop_budget,
            wall_time_limit_s=spec.wall_time_limit_s,
        )
        try:
            with budget_ctx:
                raw_predictions = estimator.predict(mlp, spec.flop_budget)
            stats = _predict_stats_to_dict(
                last_predict_stats() if callable(last_predict_stats) else None
            )
            if stats is not None:
                flops_used = int(stats.get("flops_used", budget_ctx.flops_used))
                raw_breakdown = stats.get("budget_breakdown")
            else:
                flops_used = budget_ctx.flops_used
            if raw_breakdown is None:
                raw_breakdown = budget_ctx.summary_dict(by_namespace=True)
            normalized_breakdown = _normalize_estimator_budget_breakdown(raw_breakdown)
            predictions = validate_predictions(raw_predictions, depth=spec.depth, width=spec.width)
            if normalized_breakdown is not None:
                normalized_breakdowns.append(normalized_breakdown)
        except we.BudgetExhaustedError:
            predictions = we.zeros((spec.depth, spec.width))
            budget_exhausted = True
            stats = _predict_stats_to_dict(
                last_predict_stats() if callable(last_predict_stats) else None
            )
            if stats is not None:
                flops_used = int(stats.get("flops_used", spec.flop_budget))
                raw_breakdown = stats.get("budget_breakdown")
            if raw_breakdown is None:
                raw_breakdown = budget_ctx.summary_dict(by_namespace=True)
            normalized_breakdown = _normalize_estimator_budget_breakdown(raw_breakdown)
            flops_used = flops_used or spec.flop_budget
            if normalized_breakdown is not None:
                normalized_breakdowns.append(normalized_breakdown)
        except we.TimeExhaustedError:
            predictions = we.zeros((spec.depth, spec.width))
            time_exhausted = True
            stats = _predict_stats_to_dict(
                last_predict_stats() if callable(last_predict_stats) else None
            )
            if stats is not None:
                flops_used = int(stats.get("flops_used", budget_ctx.flops_used))
                raw_breakdown = stats.get("budget_breakdown")
            if raw_breakdown is None:
                raw_breakdown = budget_ctx.summary_dict(by_namespace=True)
            normalized_breakdown = _normalize_estimator_budget_breakdown(raw_breakdown)
            if normalized_breakdown is not None:
                normalized_breakdowns.append(normalized_breakdown)
        except Exception as exc:
            predictions = we.zeros((spec.depth, spec.width))
            per_mlp.append(
                {
                    "mlp_index": i,
                    "error": str(exc),
                    "flops_used": 0,
                    "budget_exhausted": False,
                    "time_exhausted": False,
                    "untracked_time_exhausted": False,
                    "wall_time_s": 0.0,
                    "tracked_time_s": 0.0,
                    "untracked_time_s": 0.0,
                    "breakdowns": {"estimator": None},
                }
            )
            primary_scores.append(float("inf"))
            secondary_scores.append(float("inf"))
            if on_mlp_scored is not None:
                on_mlp_scored(i + 1)
            continue

        # Read timing after BudgetContext.__exit__ so wall_time_s is populated
        wall_time_s = budget_ctx.wall_time_s or 0.0
        tracked_time_s = budget_ctx.total_tracked_time
        untracked_time_s = budget_ctx.untracked_time or 0.0

        if (
            not budget_exhausted
            and not time_exhausted
            and spec.wall_time_limit_s is not None
            and wall_time_s > spec.wall_time_limit_s
        ):
            predictions = we.zeros((spec.depth, spec.width))
            time_exhausted = True

        # Post-predict check: untracked time limit
        if (
            not budget_exhausted
            and not time_exhausted
            and spec.untracked_time_limit_s is not None
            and untracked_time_s > spec.untracked_time_limit_s
        ):
            predictions = we.zeros((spec.depth, spec.width))
            untracked_time_exhausted = True

        # Convert predictions for MSE computation
        pred_np = we.asarray(predictions, dtype=we.float32)

        # Primary score: final layer MSE
        final_pred = pred_np[-1]
        final_target = data.final_targets[i]
        final_mse = float(we.mean((final_pred - final_target) ** 2))

        # Secondary score: all layers MSE
        all_target = data.all_layer_targets[i]
        all_mse = float(we.mean((pred_np - all_target) ** 2))

        primary_scores.append(final_mse)
        secondary_scores.append(all_mse)

        per_mlp.append(
            {
                "mlp_index": i,
                "final_mse": final_mse,
                "all_layer_mse": all_mse,
                "flops_used": flops_used,
                "budget_exhausted": budget_exhausted,
                "time_exhausted": time_exhausted,
                "untracked_time_exhausted": untracked_time_exhausted,
                "wall_time_s": wall_time_s,
                "tracked_time_s": tracked_time_s,
                "untracked_time_s": untracked_time_s,
                "breakdowns": {"estimator": normalized_breakdown},
            }
        )

        if on_mlp_scored is not None:
            on_mlp_scored(i + 1)

    aggregate_breakdown = _aggregate_budget_breakdowns(normalized_breakdowns)
    return {
        "primary_score": float(we.mean(we.asarray(primary_scores)))
        if primary_scores
        else float("inf"),
        "secondary_score": float(we.mean(we.asarray(secondary_scores)))
        if secondary_scores
        else float("inf"),
        "per_mlp": per_mlp,
        "breakdowns": {
            "sampling": data.sampling_budget_breakdown,
            "estimator": aggregate_breakdown,
        },
    }
