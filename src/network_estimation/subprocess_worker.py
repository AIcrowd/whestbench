"""Subprocess worker for running participant estimators in isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .domain import MLP
from .loader import load_estimator_from_path
from .sdk import BaseEstimator, SetupContext


def _payload_to_mlp(payload: dict) -> MLP:
    weights = [np.asarray(w, dtype=np.float32) for w in payload["weights"]]
    mlp = MLP(
        width=int(payload["width"]),
        depth=int(payload["depth"]),
        weights=weights,
    )
    mlp.validate()
    return mlp


def _write_response(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _handle_predict(estimator: BaseEstimator, request: dict) -> None:
    try:
        mlp = _payload_to_mlp(request["mlp"])
        budget = int(request["budget"])
    except Exception as exc:
        _write_response({"status": "error", "error_message": str(exc)})
        return

    try:
        predictions = estimator.predict(mlp, budget)
        arr = np.asarray(predictions, dtype=np.float32)
        if arr.shape != (mlp.depth, mlp.width):
            _write_response(
                {
                    "status": "error",
                    "error_message": f"Predictions shape {arr.shape} != ({mlp.depth}, {mlp.width})",
                }
            )
            return
        if not np.all(np.isfinite(arr)):
            _write_response({"status": "error", "error_message": "Non-finite predictions."})
            return
        _write_response({"status": "ok", "predictions": arr.tolist()})
    except Exception as exc:
        _write_response({"status": "error", "error_message": str(exc)})


def main() -> int:
    estimator: Optional[BaseEstimator] = None
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
                estimator, _ = load_estimator_from_path(
                    Path(entrypoint["file_path"]),
                    class_name=entrypoint.get("class_name"),
                )
                context = SetupContext(
                    width=int(ctx_payload["width"]),
                    depth=int(ctx_payload["depth"]),
                    estimator_budget=int(ctx_payload["estimator_budget"]),
                    api_version=str(ctx_payload["api_version"]),
                    scratch_dir=(
                        str(ctx_payload["scratch_dir"])
                        if ctx_payload.get("scratch_dir") is not None
                        else None
                    ),
                )
                estimator.setup(context)
                _write_response({"status": "ok"})
            except Exception as exc:
                _write_response({"status": "runtime_error", "error_message": str(exc)})
        elif command == "predict":
            if estimator is None:
                _write_response({"status": "error", "error_message": "Estimator not initialized."})
                continue
            _handle_predict(estimator, request)
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
