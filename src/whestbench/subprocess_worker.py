"""Subprocess worker for running participant estimators in isolation."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Optional

import flopscope as flops
import flopscope.numpy as fnp

from .domain import MLP
from .loader import load_estimator_from_path
from .scoring import materialise_predictions, validate_predictions
from .sdk import BaseEstimator, SetupContext


def _payload_to_mlp(payload: dict) -> MLP:
    weights = [fnp.array(fnp.asarray(w, dtype=fnp.float32)) for w in payload["weights"]]
    mlp = MLP(
        width=int(payload["width"]),
        depth=int(payload["depth"]),
        weights=weights,
        seed=int(payload.get("seed", 0)),
    )
    mlp.validate()
    return mlp


def _write_response(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _budget_payload(budget_ctx: flops.BudgetContext) -> dict:
    return {
        "flop_budget": budget_ctx.flop_budget,
        "flops_used": budget_ctx.flops_used,
        "flops_remaining": budget_ctx.flops_remaining,
        "wall_time_s": budget_ctx.wall_time_s or 0.0,
        "flopscope_backend_time_s": budget_ctx.flopscope_backend_time_s,
        "flopscope_overhead_time_s": budget_ctx.flopscope_overhead_time_s,
        "residual_wall_time_s": budget_ctx.residual_wall_time_s or 0.0,
        "by_namespace": budget_ctx.summary_dict(by_namespace=True).get("by_namespace", {}),
    }


def _handle_predict(
    estimator: BaseEstimator, request: dict, wall_time_limit_s: float | None = None
) -> None:
    try:
        mlp = _payload_to_mlp(request["mlp"])
        budget = int(request["budget"])
    except Exception as exc:
        _write_response(
            {
                "status": "error",
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return

    budget_ctx = flops.BudgetContext(
        flop_budget=budget,
        wall_time_limit_s=wall_time_limit_s,
        quiet=True,
    )
    try:
        with budget_ctx as ctx:
            predictions = estimator.predict(mlp, budget)
            # Materialise inside the window; see the note in materialise_predictions.
            # Both this and the shape check are free ops, so nothing here is billed
            # to the participant. The metered finiteness scan runs after __exit__.
            arr = materialise_predictions(predictions, depth=mlp.depth, width=mlp.width)
            # Drop our reference while the meter is still running. Otherwise the
            # object outlives the window and a __del__ finaliser runs unbilled --
            # after the response has already been written, so the flops_used we
            # just reported can no longer account for it. The local path gets this
            # for free by rebinding; here the binding would survive to function
            # scope exit. (CPython refcounting makes this immediate.)
            del predictions
            flops_used = ctx.flops_used
        arr = validate_predictions(arr, depth=mlp.depth, width=mlp.width)
        _write_response(
            {
                "status": "ok",
                "predictions": arr.tolist(),
                "flops_used": flops_used,
                "wall_time_s": budget_ctx.wall_time_s or 0.0,
                "flopscope_backend_time_s": budget_ctx.flopscope_backend_time_s,
                "flopscope_overhead_time_s": budget_ctx.flopscope_overhead_time_s,
                "residual_wall_time_s": budget_ctx.residual_wall_time_s or 0.0,
                "budget_breakdown": _budget_payload(budget_ctx),
            }
        )
    except flops.BudgetExhaustedError:
        _write_response(
            {
                "status": "budget_exhausted",
                "error_message": "FLOP budget exceeded.",
                "flops_used": budget_ctx.flops_used,
                "wall_time_s": budget_ctx.wall_time_s or 0.0,
                "flopscope_backend_time_s": budget_ctx.flopscope_backend_time_s,
                "flopscope_overhead_time_s": budget_ctx.flopscope_overhead_time_s,
                "residual_wall_time_s": budget_ctx.residual_wall_time_s or 0.0,
                "budget_breakdown": _budget_payload(budget_ctx),
            }
        )
    except flops.TimeExhaustedError:
        _write_response(
            {
                "status": "time_exhausted",
                "error_message": "Wall-clock time limit exceeded.",
                "flops_used": budget_ctx.flops_used,
                "wall_time_s": budget_ctx.wall_time_s or 0.0,
                "flopscope_backend_time_s": budget_ctx.flopscope_backend_time_s,
                "flopscope_overhead_time_s": budget_ctx.flopscope_overhead_time_s,
                "residual_wall_time_s": budget_ctx.residual_wall_time_s or 0.0,
                "budget_breakdown": _budget_payload(budget_ctx),
            }
        )
    except ValueError as exc:
        details = getattr(exc, "details", None)
        if isinstance(details, dict):
            details_payload = details
            traceback_payload = None
        else:
            details_payload = None
            traceback_payload = traceback.format_exc()
        _write_response(
            {
                "status": "error",
                "error_message": str(exc),
                "details": details_payload,
                "traceback": traceback_payload,
            }
        )
    except Exception as exc:
        _write_response(
            {
                "status": "error",
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def main() -> int:
    estimator: Optional[BaseEstimator] = None
    wall_time_limit_s: Optional[float] = None
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            _write_response({"status": "protocol_error", "error_message": "Invalid JSON."})
            continue

        command = request.get("command")
        if command == "start":
            try:
                entrypoint = request["entrypoint"]
                ctx_payload = request["context"]
                wall_time_limit_s = request.get("wall_time_limit_s")

                # Enforce memory limit before loading participant code. If the
                # platform doesn't expose RLIMIT_AS (e.g., Windows, some BSDs),
                # write a warning to stderr; the host-side LocalRunner already warns.
                memory_limit_mb = request.get("memory_limit_mb")
                if memory_limit_mb is not None and memory_limit_mb > 0:
                    try:
                        import resource as _resource

                        limit_bytes = int(memory_limit_mb) * 1024 * 1024
                        _resource.setrlimit(_resource.RLIMIT_AS, (limit_bytes, limit_bytes))
                    except (ImportError, ValueError, OSError, AttributeError) as e:
                        sys.stderr.write(
                            f"[worker] could not setrlimit RLIMIT_AS={memory_limit_mb}MB: {e}\n"
                        )

                estimator, _ = load_estimator_from_path(
                    Path(entrypoint["file_path"]),
                    class_name=entrypoint.get("class_name"),
                )
                context = SetupContext(
                    width=int(ctx_payload["width"]),
                    depth=int(ctx_payload["depth"]),
                    flop_budget=int(ctx_payload["flop_budget"]),
                    api_version=str(ctx_payload["api_version"]),
                    scratch_dir=(
                        str(ctx_payload["scratch_dir"])
                        if ctx_payload.get("scratch_dir") is not None
                        else None
                    ),
                    submission_dir=(
                        str(ctx_payload["submission_dir"])
                        if ctx_payload.get("submission_dir") is not None
                        else None
                    ),
                    seed=int(ctx_payload.get("seed", 0)),
                )
                estimator.setup(context)
                _write_response({"status": "ok"})
            except Exception as exc:
                _write_response(
                    {
                        "status": "runtime_error",
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        elif command == "predict":
            if estimator is None:
                _write_response({"status": "error", "error_message": "Estimator not initialized."})
                continue
            _handle_predict(estimator, request, wall_time_limit_s=wall_time_limit_s)
        elif command == "close":
            if estimator is not None:
                estimator.teardown()
            _write_response({"status": "ok"})
            break
        else:
            _write_response({"status": "protocol_error", "error_message": "Unknown command."})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
