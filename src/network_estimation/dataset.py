"""Create, save, and load pre-computed evaluation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint
from .simulation_backends import get_backend

SCHEMA_VERSION = "2.0"


def dataset_file_hash(path: "Path | str") -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DatasetBundle:
    metadata: Dict[str, Any]
    mlps: List[MLP]
    all_layer_means: NDArray[np.float32]
    final_means: NDArray[np.float32]
    avg_variances: List[float]

    @property
    def n_mlps(self) -> int:
        return len(self.mlps)


def create_dataset(
    *,
    n_mlps: int,
    n_samples: int,
    width: int,
    depth: int,
    estimator_budget: int,
    seed: Optional[int] = None,
    output_path: "Path | str",
    progress: Optional[Any] = None,
) -> Path:
    """Generate MLPs, compute ground truth, and save to .npz."""
    output_path = Path(output_path)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
    rng = np.random.default_rng(seed)
    backend = get_backend()

    mlps: List[MLP] = []
    for i in range(n_mlps):
        mlps.append(sample_mlp(width, depth, rng))
        if progress is not None:
            progress({"phase": "generating", "completed": i + 1, "total": n_mlps})

    # Pack weight matrices: shape (n_mlps, depth, width, width)
    weights_array = np.stack(
        [np.stack(mlp.weights) for mlp in mlps]
    ).astype(np.float32)

    # Compute ground truth
    all_means_list: List[NDArray[np.float32]] = []
    final_means_list: List[NDArray[np.float32]] = []
    avg_variances: List[float] = []
    for i, mlp in enumerate(mlps):
        all_means, final_mean, avg_var = backend.sample_layer_statistics(mlp, n_samples)
        all_means_list.append(all_means)
        final_means_list.append(final_mean)
        avg_variances.append(avg_var)
        if progress is not None:
            progress({"phase": "sampling", "completed": i + 1, "total": n_mlps})

    all_layer_means = np.stack(all_means_list).astype(np.float32)
    final_means = np.stack(final_means_list).astype(np.float32)

    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_mlps": n_mlps,
        "n_samples": n_samples,
        "width": width,
        "depth": depth,
        "estimator_budget": estimator_budget,
        "hardware": collect_hardware_fingerprint(),
    }

    np.savez(
        output_path,
        metadata=np.array(json.dumps(metadata)),
        weights=weights_array,
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=np.array(avg_variances, dtype=np.float64),
    )
    return output_path


def load_dataset(path: "Path | str") -> DatasetBundle:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))

    if "schema_version" not in metadata:
        raise ValueError("Invalid dataset: missing schema_version.")

    weights_array = data["weights"].astype(np.float32)
    all_layer_means = data["all_layer_means"].astype(np.float32)
    final_means = data["final_means"].astype(np.float32)
    avg_variances = data["avg_variances"].astype(np.float64).tolist()

    n_mlps = int(weights_array.shape[0])
    depth = int(weights_array.shape[1])
    width = int(weights_array.shape[2])

    mlps: List[MLP] = []
    for i in range(n_mlps):
        layer_weights = [weights_array[i, j] for j in range(depth)]
        mlp = MLP(width=width, depth=depth, weights=layer_weights)
        mlp.validate()
        mlps.append(mlp)

    return DatasetBundle(
        metadata=metadata,
        mlps=mlps,
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=avg_variances,
    )
