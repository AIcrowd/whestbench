"""Create, save, and load pre-computed evaluation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit, Layer
from .generation import random_circuit
from .hardware import collect_hardware_fingerprint
from .scoring import sampling_baseline_time
from .simulation import empirical_mean

SCHEMA_VERSION = "1.0"


def dataset_file_hash(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a dataset file."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """In-memory representation of a loaded evaluation dataset."""

    metadata: dict[str, Any]
    circuits: list[Circuit]
    ground_truth_means: NDArray[np.float32]
    baseline_times: NDArray[np.float64]

    @property
    def n_circuits(self) -> int:
        return len(self.circuits)


def create_dataset(
    *,
    n_circuits: int,
    n_samples: int,
    width: int,
    max_depth: int,
    budgets: list[int],
    seed: int | None = None,
    time_tolerance: float = 0.1,
    output_path: Path | str,
    progress: Any | None = None,
) -> Path:
    """Generate circuits, sample ground truth, compute baselines, and save.

    Args:
        n_circuits: Number of random circuits to generate.
        n_samples: Samples per circuit for ground truth estimation.
        width: Wire count per circuit.
        max_depth: Number of layers per circuit.
        budgets: Budget values for baseline timing.
        seed: RNG seed; auto-generated if ``None``.
        time_tolerance: Time tolerance factor stored in metadata.
        output_path: Destination for the ``.npz`` file.
        progress: Optional callback ``(dict) -> None`` for progress events.

    Returns:
        Resolved path of the saved dataset file.
    """
    output_path = Path(output_path)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
    rng = np.random.default_rng(seed)

    # --- Generate circuits ---
    circuits = [random_circuit(width, max_depth, rng) for _ in range(n_circuits)]

    # --- Pack circuit arrays ---
    circuits_first = np.stack(
        [np.stack([layer.first for layer in c.gates]) for c in circuits]
    ).astype(np.int32)
    circuits_second = np.stack(
        [np.stack([layer.second for layer in c.gates]) for c in circuits]
    ).astype(np.int32)
    circuits_coeff = np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [
                            layer.const,
                            layer.first_coeff,
                            layer.second_coeff,
                            layer.product_coeff,
                        ],
                        axis=-1,
                    )
                    for layer in c.gates
                ]
            )
            for c in circuits
        ]
    ).astype(np.float32)

    # --- Sample ground truth ---
    means_list: list[NDArray[np.float32]] = []
    for i, circuit in enumerate(circuits):
        depth_means = np.stack(list(empirical_mean(circuit, n_samples)))
        means_list.append(depth_means)
        if progress is not None:
            progress({"phase": "sampling", "completed": i + 1, "total": n_circuits})
    ground_truth_means = np.stack(means_list).astype(np.float32)

    # --- Compute baselines ---
    baseline_rows: list[list[float]] = []
    for budget in budgets:
        baseline_rows.append(sampling_baseline_time(budget, width, max_depth))
    baseline_times = np.array(baseline_rows, dtype=np.float64)

    # --- Metadata ---
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_circuits": n_circuits,
        "n_samples": n_samples,
        "width": width,
        "max_depth": max_depth,
        "budgets": budgets,
        "time_tolerance": time_tolerance,
        "hardware": collect_hardware_fingerprint(),
    }

    np.savez(
        output_path,
        metadata=np.array(json.dumps(metadata)),
        circuits_first=circuits_first,
        circuits_second=circuits_second,
        circuits_coeff=circuits_coeff,
        ground_truth_means=ground_truth_means,
        baseline_times=baseline_times,
    )
    return output_path


def load_dataset(path: Path | str) -> DatasetBundle:
    """Load a dataset bundle from a ``.npz`` file.

    Raises:
        ValueError: If the file is missing ``schema_version`` in metadata.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    metadata_raw = str(data["metadata"])
    metadata = json.loads(metadata_raw)

    if "schema_version" not in metadata:
        raise ValueError(
            "Invalid dataset file: missing 'schema_version' in metadata."
        )

    circuits_first = data["circuits_first"]
    circuits_second = data["circuits_second"]
    circuits_coeff = data["circuits_coeff"]
    ground_truth_means = data["ground_truth_means"].astype(np.float32)
    baseline_times = data["baseline_times"].astype(np.float64)

    n_circuits = int(circuits_first.shape[0])
    depth = int(circuits_first.shape[1])
    width = int(circuits_first.shape[2])

    circuits: list[Circuit] = []
    for i in range(n_circuits):
        gates: list[Layer] = []
        for j in range(depth):
            coeff = circuits_coeff[i, j]  # shape (width, 4)
            gates.append(
                Layer(
                    first=circuits_first[i, j].astype(np.int32),
                    second=circuits_second[i, j].astype(np.int32),
                    const=coeff[:, 0].astype(np.float32),
                    first_coeff=coeff[:, 1].astype(np.float32),
                    second_coeff=coeff[:, 2].astype(np.float32),
                    product_coeff=coeff[:, 3].astype(np.float32),
                )
            )
        circuit = Circuit(n=width, d=depth, gates=gates)
        circuit.validate()
        circuits.append(circuit)

    return DatasetBundle(
        metadata=metadata,
        circuits=circuits,
        ground_truth_means=ground_truth_means,
        baseline_times=baseline_times,
    )
