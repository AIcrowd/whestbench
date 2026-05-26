# pyright: reportMissingImports=false
"""Torch-backed variant of create_dataset for GPU acceleration on large bakes.

This module is a power-user drop-in alternative to whestbench.dataset.create_dataset
for n_samples >= 10^8 scenarios where the flopscope CPU path is too slow.
Torch is an optional dependency: install via `pip install whestbench[gpu]`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import platform
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset

from ._provenance import (
    flopscope_version,
    nvidia_driver_version,
    torch_determinism_state,
    whestbench_version,
)
from .dataset import _resolve_mlp_range
from .dataset_io import (
    DEFAULT_SPLIT,
    SCHEMA_FORMAT,
    SCHEMA_VERSION,
    SEED_PROTOCOL_NAME_V3,
    SEED_PROTOCOL_VERSION_V3,
    _validate_mlp_seeds,
    make_features,
    write_dataset_dir,
)
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint
from .naming import assign_unique_names


def _require_torch() -> "Any":
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "create_dataset_torch requires torch. Install with: pip install whestbench[gpu]"
        ) from exc
    return torch


def create_dataset_torch(
    *,
    n_mlps: int,
    n_samples: int,
    width: int,
    depth: int,
    mlp_seeds: Optional[List[int]] = None,
    output_path: "Path | str",
    split: str = DEFAULT_SPLIT,
    mlp_range: Optional[Tuple[int, int]] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    device: str = "auto",
    mlps_per_batch: Optional[int] = None,
    chunk_size: Optional[int] = None,
    **deprecated_kwargs: Any,
) -> Path:
    """Torch-backed analog of whestbench.dataset.create_dataset.

    Drop-in for create_dataset() at the same kwargs. Generates MLPs with the same
    seed protocol and writes a schema-3.0 Parquet+sidecar dataset directory. Output
    metadata self-identifies via backend="torch" and includes device/torch_version
    provenance.

    Statistical (not bitwise) equivalence with the flopscope CPU path holds at the
    same mlp_seeds: per-neuron means agree within ~3e-5 at N=1e9 (MC noise).

    Args:
        n_mlps, n_samples, width, depth, mlp_seeds, output_path, progress:
            Same as create_dataset(). See whestbench.dataset for full semantics.
        device: "auto" | "cuda" | "mps" | "cpu". "auto" resolves cuda > mps > cpu.
            Explicit values error if unavailable (no silent CPU fallback).
            Note: bitwise reproducibility on CUDA additionally requires the
            caller to set torch.backends.cudnn.deterministic = True. This
            function does not set that flag — on CUDA, run-to-run output is
            deterministic in practice for the matmul/sum kernels used here,
            but not formally guaranteed by torch.
        mlps_per_batch: How many MLPs to process in parallel on device.
            None (default) auto-tunes to min(n_mlps, 16).
        chunk_size: Samples per chunk on device. None (default) is memory-aware
            on cuda; fixed 65536 on mps/cpu.

    Returns:
        Path to the written dataset directory.

    Raises:
        TypeError: if the legacy ``seed=`` kwarg is passed.
        ValueError: if ``mlp_seeds`` length or values are invalid.
    """
    # Reject the legacy `seed=` kwarg with a migration hint.
    if "seed" in deprecated_kwargs:
        raise TypeError(
            "seed= is no longer supported in create_dataset_torch. "
            "Use mlp_seeds=[...] to provide explicit per-MLP seeds, "
            "or omit mlp_seeds to auto-generate them."
        )
    if deprecated_kwargs:
        unexpected = ", ".join(repr(k) for k in deprecated_kwargs)
        raise TypeError(f"create_dataset_torch() got unexpected keyword argument(s): {unexpected}")

    torch = _require_torch()

    output_path = Path(output_path)
    start, end = _resolve_mlp_range(n_mlps, mlp_range)
    resolved_device = _resolve_device(device)
    resolved_mlps_per_batch = (
        _auto_mlps_per_batch(n_mlps=end - start) if mlps_per_batch is None else int(mlps_per_batch)
    )
    resolved_chunk_size = (
        _auto_chunk_size(
            device=resolved_device, width=width, mlps_per_batch=resolved_mlps_per_batch
        )
        if chunk_size is None
        else int(chunk_size)
    )

    # Auto-generate or validate mlp_seeds.
    if mlp_seeds is None:
        # Generate distinct int63 seeds. Collisions are astronomically unlikely
        # (~n^2 / 2^64) but we re-roll defensively. The max_attempts cap prevents
        # an unbounded loop in the pathological case of a broken CSPRNG.
        seen: set = set()
        generated: List[int] = []
        max_attempts = n_mlps * 10
        for _ in range(max_attempts):
            if len(generated) >= n_mlps:
                break
            s = secrets.randbits(63)
            if s not in seen:
                seen.add(s)
                generated.append(s)
        if len(generated) < n_mlps:
            raise RuntimeError(
                f"failed to generate {n_mlps} distinct seeds in {max_attempts} attempts; "
                f"check that secrets.randbits is functioning correctly."
            )
        mlp_seeds = generated
    _validate_mlp_seeds(mlp_seeds, n_mlps)

    # Phase 1: generate MLPs on CPU (same protocol as create_dataset())
    mlps = []
    for slice_idx, i in enumerate(range(start, end)):
        ss = fnp.random.SeedSequence(mlp_seeds[i]).spawn(3)
        weight_stream = fnp.random.default_rng(ss[0])
        estimator_seed_i = int(ss[2].generate_state(1)[0])
        mlps.append(sample_mlp(width, depth, weight_stream, seed=estimator_seed_i))
        if progress is not None:
            progress({"phase": "generating", "completed": slice_idx + 1, "total": end - start})

    # Names from ALL logical estimator seeds (so slice's names equal slice of single-host bake).
    # Mirrors create_dataset() so both backends produce identical name lists at same mlp_seeds.
    all_logical_seeds = [
        int(fnp.random.SeedSequence(mlp_seeds[i]).spawn(3)[2].generate_state(1)[0])
        for i in range(n_mlps)
    ]
    all_names = assign_unique_names(all_logical_seeds)
    slice_names = all_names[start:end]
    mlps = [dataclasses.replace(m, name=n) for m, n in zip(mlps, slice_names)]

    weights_array = np.stack([np.stack(mlp.weights) for mlp in mlps]).astype(np.float32)

    # Phase 2: sampling on device, batched across MLPs
    from ._simulation_torch import sample_layer_statistics_torch

    weights_device = torch.from_numpy(weights_array).to(resolved_device)
    chunks_per_mlp = math.ceil(n_samples / resolved_chunk_size)
    slice_size = end - start
    total_sampling_chunks = slice_size * chunks_per_mlp

    all_means_list: List[np.ndarray] = []
    final_means_list: List[np.ndarray] = []
    avg_variances: List[float] = []
    sampling_budget_breakdowns: List[Dict[str, Any]] = []

    batch_starts = list(range(0, slice_size, resolved_mlps_per_batch))
    for batch_start_local in batch_starts:
        batch_end_local = min(batch_start_local + resolved_mlps_per_batch, slice_size)
        batch_size = batch_end_local - batch_start_local

        # Per-MLP torch generators seeded from the per-MLP SeedSequence stream.
        # Use logical index i to access mlp_seeds[i].spawn(3)[1] for the sample stream.
        generators = []
        for local_idx in range(batch_start_local, batch_end_local):
            i = local_idx + start  # logical index
            ss = fnp.random.SeedSequence(mlp_seeds[i]).spawn(3)
            torch_seed = int(ss[1].generate_state(1)[0])
            gen = torch.Generator(device=resolved_device)
            gen.manual_seed(torch_seed)
            generators.append(gen)

        weights_slice = weights_device[batch_start_local:batch_end_local]
        batch_names = [m.name for m in mlps[batch_start_local:batch_end_local]]

        def _on_chunk(
            event: Dict[str, Any],
            *,
            _batch_start_local: int = batch_start_local,
            batch_size_local: int = batch_size,
            batch_names_local: List[str] = batch_names,
        ) -> None:
            if progress is None:
                return
            local_completed = int(event.get("completed", 0))
            completed = _batch_start_local * chunks_per_mlp + local_completed * batch_size_local
            progress(
                {
                    "phase": "sampling",
                    "completed": completed,
                    "total": total_sampling_chunks,
                    "mlp_index_range": (
                        _batch_start_local + 1,
                        _batch_start_local + batch_size_local,
                    ),
                    "mlp_names_range": list(batch_names_local),
                    "n_mlps": end - start,
                    "unit": "chunks",
                }
            )

        wall_start = time.perf_counter()
        layer_means_batch, final_means_batch, avg_var_batch = sample_layer_statistics_torch(
            weights_batch=weights_slice,
            n_samples=n_samples,
            generators=generators,
            chunk_size=resolved_chunk_size,
            progress=_on_chunk if progress is not None else None,
        )
        wall_elapsed = time.perf_counter() - wall_start

        # Per-MLP breakdown: closed-form FLOPs + amortized wall time
        amortized_wall = wall_elapsed / batch_size
        for _ in range(batch_size):
            sampling_budget_breakdowns.append(
                _synthesize_sampling_breakdown(
                    width=width,
                    depth=depth,
                    n_samples=n_samples,
                    wall_time_s=amortized_wall,
                    flop_budget=int(1e15),
                )
            )

        layer_means_np = layer_means_batch.detach().to("cpu").numpy().astype(np.float32)
        final_means_np = final_means_batch.detach().to("cpu").numpy().astype(np.float32)
        avg_var_np = avg_var_batch.detach().to("cpu").numpy().astype(np.float64)

        for b in range(batch_size):
            all_means_list.append(layer_means_np[b])
            final_means_list.append(final_means_np[b])
            avg_variances.append(float(avg_var_np[b]))

    all_layer_means = np.stack(all_means_list).astype(np.float32)
    final_means = np.stack(final_means_list).astype(np.float32)

    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "format": SCHEMA_FORMAT,
        "backend": "torch",
        "seed_protocol": {
            "name": SEED_PROTOCOL_NAME_V3,
            "version": SEED_PROTOCOL_VERSION_V3,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_mlps": end - start,
        "n_samples": n_samples,
        "width": width,
        "depth": depth,
        "hardware": collect_hardware_fingerprint(),
        "whestbench_version": whestbench_version(),
        "flopscope_version": flopscope_version(),
        "torch_version": torch.__version__,
        "device": resolved_device,
        "mlps_per_batch": resolved_mlps_per_batch,
        "chunk_size": resolved_chunk_size,
        # Runtime state of torch's determinism levers + the cuBLAS workspace
        # env var. Bit-exact cross-host reproduction requires these to match
        # what the canonical bake used. See docs/how-to/parallel-bake.md §
        # "Bit-equivalence requirements".
        "bake_config": torch_determinism_state(),
    }
    if resolved_device == "cuda":
        metadata["cuda_device_name"] = torch.cuda.get_device_name()
        metadata["cuda_device_capability"] = list(torch.cuda.get_device_capability())
        driver = nvidia_driver_version()
        if driver is not None:
            metadata["cuda_driver_version"] = driver
    elif resolved_device == "mps":
        metadata["mps_device_name"] = platform.processor() or "Apple Silicon"

    is_partial = (start, end) != (0, n_mlps)
    if is_partial:
        metadata["is_partial"] = True
        metadata["mlp_range"] = [start, end]
        metadata["total_n_mlps"] = n_mlps

    ds = Dataset.from_dict(
        {
            "mlp_id": list(range(start, end)),
            "mlp_name": [m.name for m in mlps],
            # Under 3.0, parquet mlp_seed stores the INPUT seed (not derived estimator seed).
            "mlp_seed": mlp_seeds[start:end],
            "weights": weights_array,
            "all_layer_means": all_layer_means,
            "final_means": final_means,
            "avg_variance": avg_variances,
            "sampling_budget_breakdown": [json.dumps(b) for b in sampling_budget_breakdowns],
        },
        features=make_features(width=width, depth=depth),
    )

    write_dataset_dir(ds, output_dir=output_path, split=split, metadata=metadata)
    return output_path


# --- Internal helpers (defined at module bottom so create_dataset_torch reads top-down) ---


def _synthesize_sampling_breakdown(
    *,
    width: int,
    depth: int,
    n_samples: int,
    wall_time_s: float,
    flop_budget: int,
) -> Dict[str, Any]:
    """Closed-form analog of flopscope's BudgetContext.summary_dict for the torch path.

    The torch path computes outside flopscope's instrumentation, so this helper
    synthesizes the same dict shape using analytical FLOP counts. Verified
    against flopscope's actual count by test_closed_form_matches_flopscope_count.

    Output shape mirrors flopscope's normalized output exactly:
    - Top-level keys: flop_budget, flops_used, flops_remaining, wall_time_s,
      flopscope_backend_time_s, flopscope_overhead_time_s, residual_wall_time_s,
      by_namespace.
    - by_namespace is a FLAT dict keyed by dot-notation strings (e.g.
      "sampling.sample_layer_statistics"), NOT nested dicts.

    Formula derivation (matched exactly against flopscope's operation-level accounting):

    Per-sample ops (scale with n_samples):
      - standard_normal((n, width)):  16 FLOPs/element * n * width
        (flopscope counts RNG generation at 16 FLOPs/element, not 1)
      - array cast to fnp:             1 FLOPs/element * n * width
      - matmul (n,w)@(w,w) per layer:  n * w * (2*w - 1) per layer
        (BLAS MAC convention: w multiplies + (w-1) additions per output element;
        flopscope upgraded from `n * w^2` to this in their 2026 accounting)
      - maximum (ReLU) per layer:      n * w per layer      (1 FLOP/element)
      - sum along axis=0 per layer:    (n - 1) * w per layer
        (actual additions, not input size; flopscope upgraded from `n * w` to this)
      - power x_f64**2:               16 FLOPs/element * n * width
        (flopscope charges power at 16 FLOPs/element, unlike x*x which is 1)
      - sum for final_sum_sq:          (n - 1) * width      (actual additions)

    Per-chunk ops (scale with n_chunks):
      - add layer_sums += per layer:   width per layer per chunk
      - add final_sum_sq += :          width per chunk

    Post-loop ops (once):
      - true_divide layer_sums/n:      depth * width
      - true_divide final_sum_sq/n:    width
      - power final_mean**2:           16 * width           (same 16 FLOPs/element rule)
      - subtract (sq/n - mean^2):      width
      - mean (avg variance):           width
    """
    # chunk_size mirrors simulation._pick_chunk_size
    chunk_size = max(1024, min(16384, 2**20 // width))
    n_chunks = math.ceil(n_samples / chunk_size)

    # Per-sample costs
    standard_normal = 16 * n_samples * width
    array_cast = n_samples * width
    # BLAS MAC convention: (n, w) @ (w, w) costs n * w * (2*w - 1) per call.
    matmul = depth * n_samples * width * (2 * width - 1)
    relu = depth * n_samples * width
    # sum along axis=0 on (n, w): (n - 1) actual additions per output column.
    sum_layer = depth * (n_samples - 1) * width
    power_sq = 16 * n_samples * width
    sum_sq = (n_samples - 1) * width

    # Per-chunk costs
    add_costs = n_chunks * (depth * width + width)

    # Post-loop costs
    true_divide_layer = depth * width
    true_divide_sq = width
    power_mean = 16 * width
    subtract = width
    mean_reduction = width

    total = (
        standard_normal
        + array_cast
        + matmul
        + relu
        + sum_layer
        + power_sq
        + sum_sq
        + add_costs
        + true_divide_layer
        + true_divide_sq
        + power_mean
        + subtract
        + mean_reduction
    )

    flops_remaining = max(0, flop_budget - total)

    return {
        "flop_budget": flop_budget,
        "flops_used": total,
        "flops_remaining": flops_remaining,
        "wall_time_s": wall_time_s,
        "flopscope_backend_time_s": 0.0,
        "flopscope_overhead_time_s": 0.0,
        "residual_wall_time_s": wall_time_s,
        "by_namespace": {
            "sampling.sample_layer_statistics": {
                "flops_used": total,
                "calls": 0,
                "flopscope_backend_time_s": 0.0,
                "flopscope_overhead_time_s": 0.0,
                "operations": {},
            }
        },
    }


def _resolve_device(device: str) -> str:
    """Resolve a user-facing device string to a concrete torch device kind.

    Args:
        device: One of "auto", "cuda", "mps", "cpu".

    Returns:
        A concrete device kind: "cuda", "mps", or "cpu". Never "auto".

    Raises:
        ValueError: If device is not one of the accepted values.
        RuntimeError: If an explicit device is requested but unavailable.
    """
    import torch  # local import: torch is an optional dep

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Either CUDA is not installed, or torch was built without "
                "CUDA support. For dev without a GPU, use device='cpu'. "
                "Install: pip install whestbench[gpu]"
            )
        return "cuda"
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested but torch.backends.mps.is_available() is False. "
                "MPS is only supported on Apple Silicon with macOS 12.3+. "
                "For dev elsewhere, use device='cpu'."
            )
        return "mps"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"device must be one of 'auto', 'cuda', 'mps', 'cpu'; got {device!r}")


def _auto_mlps_per_batch(*, n_mlps: int) -> int:
    """Default mlps_per_batch: cap at 16 to bound GPU memory growth."""
    return min(n_mlps, 16)


def _auto_chunk_size(*, device: str, width: int, mlps_per_batch: int) -> int:
    """Default chunk_size.

    On cuda: targets ~25% of free GPU memory for the activations tensor,
    clamped to [65536, 1<<20]. On mps/cpu: fixed 65536 (good balance of
    kernel-launch amortization and memory).
    """
    if device != "cuda":
        return 65536
    import torch  # local

    free_bytes, _ = torch.cuda.mem_get_info()
    target_bytes = min(2 * 1024**3, free_bytes // 4)
    size = target_bytes // (mlps_per_batch * width * 4)
    return max(65536, min(1 << 20, int(size)))
