"""Subprocess worker entrypoint for running participant estimators in isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .domain import Circuit, Layer
from .loader import load_estimator_from_path
from .sdk import BaseEstimator, SetupContext


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
                    {"status": "protocol_error", "error_message": "Estimator is not initialized."}
                )
                continue
            try:
                circuit = _payload_to_circuit(request["circuit"])
                budget = int(request["budget"])
                predictions = estimator.predict(circuit, budget)
                if not isinstance(predictions, np.ndarray):
                    _write_response(
                        {
                            "status": "protocol_error",
                            "error_message": "Estimator predict must return numpy.ndarray.",
                        }
                    )
                    continue
                _write_response({"status": "ok", "predictions": predictions.tolist()})
            except Exception as exc:
                _write_response({"status": "runtime_error", "error_message": str(exc)})
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
