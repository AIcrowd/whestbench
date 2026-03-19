"""Subprocess worker entrypoint for running participant estimators in isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

from .domain import Circuit, Layer
from .loader import load_estimator_from_path
from .sdk import BaseEstimator, SetupContext
from .streaming import validate_depth_row


def _payload_to_circuit(payload: dict[str, Any]) -> Circuit:
    gates: list[Layer] = []
    for gate_payload in payload["gates"]:
        gates.append(
            Layer(
                first=np.asarray(gate_payload["first"], dtype=np.int32),
                second=np.asarray(gate_payload["second"], dtype=np.int32),
                first_coeff=np.asarray(gate_payload["first_coeff"], dtype=np.float32),
                second_coeff=np.asarray(gate_payload["second_coeff"], dtype=np.float32),
                const=np.asarray(gate_payload["const"], dtype=np.float32),
                product_coeff=np.asarray(gate_payload["product_coeff"], dtype=np.float32),
            )
        )
    circuit = Circuit(n=int(payload["n"]), d=int(payload["d"]), gates=gates)
    circuit.validate()
    return circuit


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _handle_predict(estimator: BaseEstimator, request: dict[str, Any]) -> None:
    """Stream one JSON line per depth row back to the runner."""
    try:
        circuit = _payload_to_circuit(request["circuit"])
        budget = int(request["budget"])
    except Exception as exc:
        _write_response({"status": "error", "depth_index": 0, "error_message": str(exc)})
        _write_response({"status": "done"})
        return

    try:
        raw_predictions = estimator.predict(circuit, budget)
    except Exception as exc:
        _write_response({"status": "error", "depth_index": 0, "error_message": str(exc)})
        _write_response({"status": "done"})
        return

    try:
        output_iter = iter(cast(Any, raw_predictions))
    except TypeError:
        _write_response({
            "status": "error",
            "depth_index": 0,
            "error_message": "Estimator must return an iterator of depth-row outputs.",
        })
        _write_response({"status": "done"})
        return

    for depth_index in range(circuit.d):
        try:
            raw_row = next(output_iter)
        except StopIteration:
            _write_response({
                "status": "error",
                "depth_index": depth_index,
                "error_message": "Estimator must emit exactly max_depth rows.",
            })
            _write_response({"status": "done"})
            return
        except Exception as exc:
            _write_response({
                "status": "error",
                "depth_index": depth_index,
                "error_message": f"Estimator stream failed at depth {depth_index}: {exc}",
            })
            _write_response({"status": "done"})
            return

        try:
            row = validate_depth_row(raw_row, width=circuit.n, depth_index=depth_index)
        except ValueError as exc:
            _write_response({
                "status": "error",
                "depth_index": depth_index,
                "error_message": str(exc),
            })
            _write_response({"status": "done"})
            return

        _write_response({
            "status": "row",
            "depth_index": depth_index,
            "row": row.tolist(),
        })

    _write_response({"status": "done"})


def main() -> int:
    estimator: BaseEstimator | None = None
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            _write_response(
                {
                    "status": "protocol_error",
                    "error_message": "Invalid JSON request payload.",
                }
            )
            continue

        command = request.get("command")
        if command == "start":
            try:
                entrypoint = request["entrypoint"]
                context_payload = request["context"]
                estimator, _ = load_estimator_from_path(
                    Path(entrypoint["file_path"]),
                    class_name=entrypoint.get("class_name"),
                )
                context = SetupContext(
                    width=int(context_payload["width"]),
                    max_depth=int(context_payload["max_depth"]),
                    budgets=tuple(int(b) for b in context_payload["budgets"]),
                    time_tolerance=float(context_payload["time_tolerance"]),
                    api_version=str(context_payload["api_version"]),
                    scratch_dir=(
                        str(context_payload["scratch_dir"])
                        if context_payload.get("scratch_dir") is not None
                        else None
                    ),
                )
                estimator.setup(context)
                _write_response({"status": "ok"})
            except Exception as exc:  # pragma: no cover - exercised via integration tests
                _write_response({"status": "runtime_error", "error_message": str(exc)})
        elif command == "predict":
            if estimator is None:
                _write_response(
                    {"status": "error", "depth_index": 0, "error_message": "Estimator not initialized."}
                )
                _write_response({"status": "done"})
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
