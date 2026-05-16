"""Scoring loop and FLOP budget enforcement for MLP estimation contests."""

from __future__ import annotations

import traceback as _tb
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import flopscope as flops
import flopscope.numpy as fnp

from .domain import MLP
from .generation import sample_mlp
from .runner import RunnerError
from .sdk import BaseEstimator
from .simulation import sample_layer_statistics, sample_layer_statistics_chunk_count

if TYPE_CHECKING:
    from .dataset import DatasetBundle

# FLOP-equivalent rate for residual wall time. Used in budget-adjusted scoring:
#   effective compute C_m = F_m + LAMBDA_FLOPS_PER_SECOND * R_m
# Where R_m is the residual wall-time bucket (NOT total wall, NOT flopscope dispatch).
# Per the NeurIPS proposal, the initial rate is 10^11 FLOPs/second.
LAMBDA_FLOPS_PER_SECOND: float = 1e11


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
    residual_wall_time_limit_s: Optional[float] = None
    seed: Optional[int] = None

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
        if self.residual_wall_time_limit_s is not None and self.residual_wall_time_limit_s <= 0:
            raise ValueError("residual_wall_time_limit_s must be positive when provided.")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an integer when provided.")


@dataclass
class ContestData:
    """Precomputed contest data for scoring."""

    spec: ContestSpec
    mlps: List[MLP]
    all_layer_targets: List[fnp.ndarray]
    final_targets: List[fnp.ndarray]
    avg_variances: List[float]
    sampling_budget_breakdown: Optional[Dict[str, Any]] = None


def make_contest(
    spec: ContestSpec,
    on_mlp_done: Optional[Callable[[int], None]] = None,
    *,
    on_sampling_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> ContestData:
    """Generate MLPs, compute ground truth, and collect sampling attribution."""
    spec.validate()
    spec_seed = spec.seed
    stream_seeds: List[fnp.random.SeedSequence] = (
        fnp.random.SeedSequence(int(spec_seed)).spawn(3 * spec.n_mlps)
        if spec_seed is not None
        else []
    )

    mlps: List[MLP] = []
    all_layer_targets: List[fnp.ndarray] = []
    final_targets: List[fnp.ndarray] = []
    avg_variances: List[float] = []
    sampling_breakdowns: List[Dict[str, Any]] = []
    chunks_per_mlp = sample_layer_statistics_chunk_count(spec.width, spec.ground_truth_samples)
    total_chunks = spec.n_mlps * chunks_per_mlp

    for i in range(spec.n_mlps):
        mlp_rng = fnp.random.default_rng(stream_seeds[3 * i]) if spec_seed is not None else None
        sample_rng = (
            fnp.random.default_rng(stream_seeds[3 * i + 1]) if spec_seed is not None else None
        )
        estimator_seed = (
            int(stream_seeds[3 * i + 2].generate_state(1)[0]) if spec_seed is not None else 0
        )
        mlp = sample_mlp(spec.width, spec.depth, mlp_rng, seed=estimator_seed)

        def _on_sampling_chunk(
            event: Dict[str, Any], *, mlp_index: int = i + 1, chunk_offset: int = i * chunks_per_mlp
        ) -> None:
            if on_sampling_progress is None:
                return
            local_completed = int(event.get("completed", 0))
            local_total = int(event.get("total", chunks_per_mlp))
            on_sampling_progress(
                {
                    "phase": "sampling_ground_truth",
                    "completed": chunk_offset + local_completed,
                    "total": total_chunks,
                    "mlp_index": mlp_index,
                    "n_mlps": spec.n_mlps,
                    "mlp_completed": local_completed,
                    "mlp_total": local_total,
                    "unit": "chunks",
                }
            )

        with flops.BudgetContext(flop_budget=int(1e15), quiet=True) as sampling_budget:
            with flops.namespace("sampling"):
                with flops.namespace("sample_layer_statistics"):
                    all_means, final_mean, avg_var = sample_layer_statistics(
                        mlp,
                        spec.ground_truth_samples,
                        rng=sample_rng,
                        progress=_on_sampling_chunk if on_sampling_progress is not None else None,
                    )
        normalized_sampling = _normalize_sampling_budget_breakdown(
            sampling_budget.summary_dict(by_namespace=True)
        )
        if normalized_sampling is not None:
            sampling_breakdowns.append(normalized_sampling)
        # Convert to flopscope arrays for ground truth storage.
        # The triple is always bound: flopscope's namespace `__exit__` returns
        # False at runtime (never suppresses), but pyright reads the `-> bool`
        # annotation conservatively.
        all_layer_targets.append(fnp.asarray(all_means, dtype=fnp.float32))  # pyright: ignore[reportPossiblyUnboundVariable]
        final_targets.append(fnp.asarray(final_mean, dtype=fnp.float32))  # pyright: ignore[reportPossiblyUnboundVariable]
        mlps.append(mlp)
        avg_variances.append(avg_var)  # pyright: ignore[reportPossiblyUnboundVariable]
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
    Ground truth is not recomputed. Sampling attribution is restored from the
    dataset when available and aggregated across the first ``n_mlps`` entries.

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
        fnp.asarray(bundle.all_layer_means[i], dtype=fnp.float32) for i in range(n_mlps)
    ]
    final_targets = [fnp.asarray(bundle.final_means[i], dtype=fnp.float32) for i in range(n_mlps)]
    avg_variances = list(bundle.avg_variances[:n_mlps])

    return ContestData(
        spec=spec,
        mlps=mlps,
        all_layer_targets=all_layer_targets,
        final_targets=final_targets,
        avg_variances=avg_variances,
        sampling_budget_breakdown=_aggregate_budget_breakdowns(
            list(bundle.sampling_budget_breakdowns[:n_mlps])
            if isinstance(bundle.sampling_budget_breakdowns, list)
            else []
        ),
    )


def validate_predictions(predictions: fnp.ndarray, *, depth: int, width: int) -> fnp.ndarray:
    """Validate estimator prediction array shape and finiteness."""
    shape = tuple(predictions.shape) if hasattr(predictions, "shape") else ()
    expected_shape = (depth, width)
    if shape != expected_shape:
        hint = (
            "Returned predictions appear to be transposed: expected (depth, width), got (width, depth)."
            if shape == (width, depth)
            else "Predictions must be a 2D array with shape (depth, width)."
        )
        details = {
            "expected_shape": list(expected_shape),
            "got_shape": list(shape),
            "cause_hints": [hint],
            "hint": hint,
        }
        exc = ValueError(f"Predictions must have shape ({depth}, {width}), got {shape}.")
        setattr(exc, "details", details)
        raise exc
    pred_np = fnp.asarray(predictions, dtype=fnp.float32)
    if not fnp.all(fnp.isfinite(pred_np)):
        details = {
            "expected_shape": list(expected_shape),
            "got_shape": list(shape),
            "cause_hints": ["Predictions must contain finite values only."],
            "hint": "Prediction values must be finite and include neither inf nor NaN.",
        }
        exc = ValueError("Predictions must contain only finite values.")
        setattr(exc, "details", details)
        raise exc
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
        "flopscope_backend_time_s",
        "flopscope_overhead_time_s",
        "residual_wall_time_s",
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


def _empty_operation_timing() -> Dict[str, Any]:
    return {
        "flop_cost": 0,
        "calls": 0,
        "flopscope_backend_time_s": 0.0,
        "flopscope_overhead_time_s": 0.0,
    }


def _merge_operation_timing(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["flop_cost"] += int(source.get("flop_cost", 0))
    target["calls"] += int(source.get("calls", 0))
    target["flopscope_backend_time_s"] += float(source.get("flopscope_backend_time_s", 0.0) or 0.0)
    target["flopscope_overhead_time_s"] += float(
        source.get("flopscope_overhead_time_s", 0.0) or 0.0
    )


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
        "flopscope_backend_time_s": float(raw.get("flopscope_backend_time_s", 0.0) or 0.0),
        "flopscope_overhead_time_s": float(raw["flopscope_overhead_time_s"]),
        "residual_wall_time_s": float(raw.get("residual_wall_time_s", 0.0) or 0.0),
        "by_namespace": {},
    }
    by_namespace = raw.get("by_namespace") or {}
    for namespace, bucket in by_namespace.items():
        normalized_namespace = _normalize_sampling_namespace(namespace)
        merged_bucket = normalized["by_namespace"].setdefault(
            normalized_namespace,
            {
                "flops_used": 0,
                "calls": 0,
                "flopscope_backend_time_s": 0.0,
                "flopscope_overhead_time_s": 0.0,
                "operations": {},
            },
        )
        merged_bucket["flops_used"] += int(bucket.get("flops_used", 0))
        merged_bucket["calls"] += int(bucket.get("calls", 0))
        merged_bucket["flopscope_backend_time_s"] += float(
            bucket.get("flopscope_backend_time_s", 0.0) or 0.0
        )
        merged_bucket["flopscope_overhead_time_s"] += float(bucket["flopscope_overhead_time_s"])
        for op_name, op_info in (bucket.get("operations") or {}).items():
            merged_op = merged_bucket["operations"].setdefault(op_name, _empty_operation_timing())
            _merge_operation_timing(merged_op, op_info)
    if not normalized["by_namespace"] and normalized["flops_used"] > 0:
        normalized["by_namespace"]["sampling.sample_layer_statistics"] = {
            "flops_used": normalized["flops_used"],
            "calls": int(raw.get("calls", 0)),
            "flopscope_backend_time_s": normalized["flopscope_backend_time_s"],
            "flopscope_overhead_time_s": normalized["flopscope_overhead_time_s"],
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
        "flopscope_backend_time_s": float(raw.get("flopscope_backend_time_s", 0.0) or 0.0),
        "flopscope_overhead_time_s": float(raw["flopscope_overhead_time_s"]),
        "residual_wall_time_s": float(raw.get("residual_wall_time_s", 0.0) or 0.0),
        "by_namespace": {},
    }
    by_namespace = raw.get("by_namespace") or {}
    for namespace, bucket in by_namespace.items():
        normalized_namespace = _normalize_estimator_namespace(namespace)
        merged_bucket = normalized["by_namespace"].setdefault(
            normalized_namespace,
            {
                "flops_used": 0,
                "calls": 0,
                "flopscope_backend_time_s": 0.0,
                "flopscope_overhead_time_s": 0.0,
                "operations": {},
            },
        )
        merged_bucket["flops_used"] += int(bucket.get("flops_used", 0))
        merged_bucket["calls"] += int(bucket.get("calls", 0))
        merged_bucket["flopscope_backend_time_s"] += float(
            bucket.get("flopscope_backend_time_s", 0.0) or 0.0
        )
        merged_bucket["flopscope_overhead_time_s"] += float(bucket["flopscope_overhead_time_s"])
        for op_name, op_info in (bucket.get("operations") or {}).items():
            merged_op = merged_bucket["operations"].setdefault(op_name, _empty_operation_timing())
            _merge_operation_timing(merged_op, op_info)
    if not normalized["by_namespace"] and normalized["flops_used"] > 0:
        normalized["by_namespace"]["estimator.estimator-client"] = {
            "flops_used": normalized["flops_used"],
            "calls": int(raw.get("calls", 0)),
            "flopscope_backend_time_s": normalized["flopscope_backend_time_s"],
            "flopscope_overhead_time_s": normalized["flopscope_overhead_time_s"],
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
        "flopscope_backend_time_s": 0.0,
        "flopscope_overhead_time_s": 0.0,
        "residual_wall_time_s": 0.0,
        "by_namespace": {},
    }
    for breakdown in breakdowns:
        aggregate["flop_budget"] += int(breakdown.get("flop_budget", 0))
        aggregate["flops_used"] += int(breakdown.get("flops_used", 0))
        aggregate["flops_remaining"] += int(breakdown.get("flops_remaining", 0))
        aggregate["wall_time_s"] += float(breakdown.get("wall_time_s", 0.0) or 0.0)
        aggregate["flopscope_backend_time_s"] += float(
            breakdown.get("flopscope_backend_time_s", 0.0) or 0.0
        )
        aggregate["flopscope_overhead_time_s"] += float(breakdown["flopscope_overhead_time_s"])
        aggregate["residual_wall_time_s"] += float(
            breakdown.get("residual_wall_time_s", 0.0) or 0.0
        )
        for namespace, bucket in (breakdown.get("by_namespace") or {}).items():
            merged = aggregate["by_namespace"].setdefault(
                namespace,
                {
                    "flops_used": 0,
                    "calls": 0,
                    "flopscope_backend_time_s": 0.0,
                    "flopscope_overhead_time_s": 0.0,
                    "operations": {},
                },
            )
            merged["flops_used"] += int(bucket.get("flops_used", 0))
            merged["calls"] += int(bucket.get("calls", 0))
            merged["flopscope_backend_time_s"] += float(
                bucket.get("flopscope_backend_time_s", 0.0) or 0.0
            )
            merged["flopscope_overhead_time_s"] += float(bucket["flopscope_overhead_time_s"])
            for op_name, op_info in (bucket.get("operations") or {}).items():
                op_bucket = merged["operations"].setdefault(op_name, _empty_operation_timing())
                _merge_operation_timing(op_bucket, op_info)
    return aggregate


def _compute_budget_adjusted_score(
    *, mse_final: float, effective_compute: float, flop_budget: int, failure: bool
) -> float:
    """Compute the per-MLP budget-adjusted score s_m.

    For valid runs: s_m = mse_final * max(0.5, C_m / B_m).
    For failures: s_m = mse_final * 1.0 (no compute discount).

    The 0.5 floor on the success-path multiplier prevents an arbitrarily
    cheap submission from dominating the ranking. The 1.0 multiplier on
    failures ensures that a failed run is strictly worse than a trivial-zero
    submission that succeeds (which gets the 0.5 floor).
    """
    if failure:
        return float(mse_final)
    if flop_budget <= 0:
        return float(mse_final)
    ratio = effective_compute / float(flop_budget)
    multiplier = max(0.5, ratio)
    return float(mse_final) * multiplier


def evaluate_estimator(
    estimator: BaseEstimator,
    data: ContestData,
    on_mlp_scored: Optional[Callable[[int], None]] = None,
    *,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """Score an estimator against precomputed contest data.

    Each MLP prediction runs under a BudgetContext. If the FLOP budget,
    wall-time limit, or residual wall-time limit is exceeded, predictions are
    zeroed and the violation is recorded. Score per MLP = final_layer_mse *
    max(0.5, C_m / B_m) for valid runs, where C_m = F_m + lambda * R_m. For
    failure-flagged runs (FLOP/time/residual budget exhausted), the multiplier
    is forced to 1.0. The aggregate adjusted_final_layer_mse is the arithmetic
    mean of per-MLP scores. Lower is better.

    When ``fail_fast`` is True, unexpected predict-time exceptions are re-raised
    instead of being recorded and skipped.
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
        residual_wall_time_exhausted = False
        raw_breakdown: Optional[Dict[str, Any]] = None
        normalized_breakdown: Optional[Dict[str, Any]] = None

        budget_ctx = flops.BudgetContext(
            flop_budget=spec.flop_budget,
            wall_time_limit_s=spec.wall_time_limit_s,
            quiet=True,
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
        except flops.BudgetExhaustedError:
            predictions = fnp.zeros((spec.depth, spec.width))
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
        except flops.TimeExhaustedError:
            predictions = fnp.zeros((spec.depth, spec.width))
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
            if fail_fast:
                raise
            predictions = fnp.zeros((spec.depth, spec.width))
            # Prefer the traceback forwarded from a remote runner (subprocess
            # worker) when available; otherwise capture the local chain.
            tb_text: Optional[str] = None
            if isinstance(exc, RunnerError) and exc.detail.traceback:
                tb_text = exc.detail.traceback
            else:
                tb_text = _tb.format_exc()
            error_message: object
            if isinstance(exc, RunnerError):
                error_code = exc.detail.code
                if exc.detail.details is not None:
                    error_message = {"message": str(exc), "details": exc.detail.details}
                else:
                    error_message = str(exc)
            elif isinstance(getattr(exc, "details", None), dict):
                error_code = exc.__class__.__name__
                error_message = {"message": str(exc), "details": getattr(exc, "details")}
            else:
                error_code = exc.__class__.__name__
                error_message = str(exc)

            # Compute zero-prediction MSE against this MLP's targets.
            pred_np = fnp.asarray(predictions, dtype=fnp.float32)
            final_target = data.final_targets[i]
            all_target = data.all_layer_targets[i]
            final_layer_mse_fail = float(fnp.mean((pred_np[-1] - final_target) ** 2))
            all_layers_mse_fail = float(fnp.mean((pred_np - all_target) ** 2))
            s_m_fail = _compute_budget_adjusted_score(
                mse_final=final_layer_mse_fail,
                effective_compute=0.0,
                flop_budget=spec.flop_budget,
                failure=True,
            )

            per_mlp.append(
                {
                    "mlp_index": i,
                    "error": error_message,
                    "error_code": error_code,
                    "traceback": tb_text,
                    "final_layer_mse": final_layer_mse_fail,
                    "all_layers_mse": all_layers_mse_fail,
                    "adjusted_final_layer_mse": s_m_fail,
                    "flops_used": 0,
                    "effective_compute": 0.0,
                    "budget_exhausted": False,
                    "time_exhausted": False,
                    "residual_wall_time_exhausted": False,
                    "combined_budget_exhausted": False,
                    "wall_time_s": 0.0,
                    "flopscope_backend_time_s": 0.0,
                    "flopscope_overhead_time_s": 0.0,
                    "residual_wall_time_s": 0.0,
                    "breakdowns": {"estimator": None},
                }
            )
            primary_scores.append(s_m_fail)
            secondary_scores.append(all_layers_mse_fail)
            if on_mlp_scored is not None:
                on_mlp_scored(i + 1)
            continue

        # Read timing: prefer worker-reported stats (subprocess runner) over
        # host budget_ctx (local runner). Under subprocess, the host's budget_ctx
        # only wraps the IPC dispatch and its residual is dominated by JSON/pipe
        # I/O, unrelated to the participant's actual residual time. The worker
        # measures its own timing inside its own BudgetContext and reports via
        # JSON; last_predict_stats() surfaces that on the runner-wrapped estimator.
        # Under --runner local, last_predict_stats() returns None and we fall
        # back to budget_ctx values, which are correct because the estimator
        # ran in the same address space.
        stats = _predict_stats_to_dict(
            last_predict_stats() if callable(last_predict_stats) else None
        )
        if stats is not None:
            wall_time_s = float(stats.get("wall_time_s", budget_ctx.wall_time_s or 0.0) or 0.0)
            flopscope_backend_time_s = float(
                stats.get("flopscope_backend_time_s", budget_ctx.flopscope_backend_time)
            )
            flopscope_overhead_time_s = float(
                stats.get("flopscope_overhead_time_s", budget_ctx.flopscope_overhead_time)
            )
            residual_wall_time_s = float(
                stats.get("residual_wall_time_s", budget_ctx.residual_wall_time or 0.0) or 0.0
            )
        else:
            wall_time_s = budget_ctx.wall_time_s or 0.0
            flopscope_backend_time_s = budget_ctx.flopscope_backend_time
            flopscope_overhead_time_s = budget_ctx.flopscope_overhead_time
            residual_wall_time_s = budget_ctx.residual_wall_time or 0.0

        if (
            not budget_exhausted
            and not time_exhausted
            and spec.wall_time_limit_s is not None
            and wall_time_s > spec.wall_time_limit_s
        ):
            predictions = fnp.zeros((spec.depth, spec.width))
            time_exhausted = True

        # Post-predict check: residual wall-time limit
        if (
            not budget_exhausted
            and not time_exhausted
            and spec.residual_wall_time_limit_s is not None
            and residual_wall_time_s > spec.residual_wall_time_limit_s
        ):
            predictions = fnp.zeros((spec.depth, spec.width))
            residual_wall_time_exhausted = True

        effective_compute = float(flops_used) + LAMBDA_FLOPS_PER_SECOND * float(
            residual_wall_time_s
        )

        # Post-hoc combined-budget check: C_m = F_m + lambda * R_m > B_m → zero out.
        # The individual caps (flop_budget for F_m, residual_wall_time_limit_s for R_m)
        # are generous on each axis; this catches the worst case where a participant
        # uses near-B_m on both. Bounded waste of grader compute (up to ~2 B_m), with
        # the participant scored as failure regardless.
        combined_budget_exhausted = False
        if (
            not budget_exhausted
            and not time_exhausted
            and not residual_wall_time_exhausted
            and effective_compute > spec.flop_budget
        ):
            predictions = fnp.zeros((spec.depth, spec.width))
            combined_budget_exhausted = True

        # Convert predictions for MSE computation
        pred_np = fnp.asarray(predictions, dtype=fnp.float32)

        # Primary score: final layer MSE
        final_pred = pred_np[-1]
        final_target = data.final_targets[i]
        final_layer_mse = float(fnp.mean((final_pred - final_target) ** 2))

        # Secondary score: all layers MSE
        all_target = data.all_layer_targets[i]
        all_layers_mse = float(fnp.mean((pred_np - all_target) ** 2))

        # Budget-adjusted per-MLP score:
        #   s_m = final_layer_mse * max(0.5, C_m / B_m)  for valid runs
        #   s_m = final_layer_mse * 1.0                  for failures (Task 5 wires this for exceptions)
        failure_flag = (
            budget_exhausted
            or time_exhausted
            or residual_wall_time_exhausted
            or combined_budget_exhausted
        )
        adjusted_final_layer_mse = _compute_budget_adjusted_score(
            mse_final=final_layer_mse,
            effective_compute=effective_compute,
            flop_budget=spec.flop_budget,
            failure=failure_flag,
        )

        primary_scores.append(adjusted_final_layer_mse)
        secondary_scores.append(all_layers_mse)

        per_mlp.append(
            {
                "mlp_index": i,
                "final_layer_mse": final_layer_mse,
                "all_layers_mse": all_layers_mse,
                "adjusted_final_layer_mse": adjusted_final_layer_mse,
                "flops_used": flops_used,
                "effective_compute": effective_compute,
                "budget_exhausted": budget_exhausted,
                "time_exhausted": time_exhausted,
                "residual_wall_time_exhausted": residual_wall_time_exhausted,
                "combined_budget_exhausted": combined_budget_exhausted,
                "wall_time_s": wall_time_s,
                "flopscope_backend_time_s": flopscope_backend_time_s,
                "flopscope_overhead_time_s": flopscope_overhead_time_s,
                "residual_wall_time_s": residual_wall_time_s,
                "breakdowns": {"estimator": normalized_breakdown},
            }
        )

        if on_mlp_scored is not None:
            on_mlp_scored(i + 1)

    aggregate_breakdown = _aggregate_budget_breakdowns(normalized_breakdowns)
    # Aggregate the new raw final-layer MSE list across MLPs.
    raw_final_layer_mses = [
        float(entry["final_layer_mse"])
        for entry in per_mlp
        if isinstance(entry, dict) and "final_layer_mse" in entry
    ]

    # Per-MLP value lists for new aggregates.
    adjusted_values = [
        float(entry["adjusted_final_layer_mse"])
        for entry in per_mlp
        if isinstance(entry, dict) and "adjusted_final_layer_mse" in entry
    ]
    effective_computes = [
        float(entry.get("effective_compute", 0.0)) for entry in per_mlp if isinstance(entry, dict)
    ]

    # Per-MLP multiplier: 1.0 on failure; else max(0.5, C_m / B_m).
    flag_keys = (
        "budget_exhausted",
        "time_exhausted",
        "residual_wall_time_exhausted",
        "combined_budget_exhausted",
    )

    def _is_failed_entry(e: Dict[str, Any]) -> bool:
        return bool(e.get("error_code")) or any(bool(e.get(k)) for k in flag_keys)

    multipliers: List[float] = []
    utilizations: List[float] = []
    for entry in per_mlp:
        if not isinstance(entry, dict):
            continue
        b = spec.flop_budget
        cm = float(entry.get("effective_compute", 0.0))
        u = (cm / float(b)) if b > 0 else 0.0
        utilizations.append(u)
        if _is_failed_entry(entry):
            multipliers.append(1.0)
        else:
            multipliers.append(max(0.5, u))

    n_failed = sum(1 for entry in per_mlp if isinstance(entry, dict) and _is_failed_entry(entry))

    failure_breakdown = {
        "budget_exhausted": sum(
            1 for e in per_mlp if isinstance(e, dict) and bool(e.get("budget_exhausted"))
        ),
        "time_exhausted": sum(
            1 for e in per_mlp if isinstance(e, dict) and bool(e.get("time_exhausted"))
        ),
        "residual_wall_time_exhausted": sum(
            1
            for e in per_mlp
            if isinstance(e, dict) and bool(e.get("residual_wall_time_exhausted"))
        ),
        "combined_budget_exhausted": sum(
            1 for e in per_mlp if isinstance(e, dict) and bool(e.get("combined_budget_exhausted"))
        ),
        "error": sum(1 for e in per_mlp if isinstance(e, dict) and bool(e.get("error_code"))),
    }

    return {
        "adjusted_final_layer_mse": float(fnp.mean(fnp.asarray(primary_scores)))
        if primary_scores
        else float("inf"),
        "final_layer_mse": float(fnp.mean(fnp.asarray(raw_final_layer_mses)))
        if raw_final_layer_mses
        else float("inf"),
        "all_layers_mse": float(fnp.mean(fnp.asarray(secondary_scores)))
        if secondary_scores
        else float("inf"),
        "best_mlp_adjusted_final_layer_mse": min(adjusted_values)
        if adjusted_values
        else float("inf"),
        "worst_mlp_adjusted_final_layer_mse": max(adjusted_values)
        if adjusted_values
        else float("inf"),
        "mean_score_multiplier": (sum(multipliers) / len(multipliers)) if multipliers else 1.0,
        "mean_compute_utilization": (sum(utilizations) / len(utilizations))
        if utilizations
        else 0.0,
        "n_failed_mlps": n_failed,
        "mean_effective_compute": (sum(effective_computes) / len(effective_computes))
        if effective_computes
        else 0.0,
        "failure_breakdown": failure_breakdown,
        "per_mlp": per_mlp,
        "breakdowns": {
            "sampling": data.sampling_budget_breakdown,
            "estimator": aggregate_breakdown,
        },
    }
