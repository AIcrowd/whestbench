"""Torch-backed variant of create_dataset for GPU acceleration on large bakes.

This module is a power-user drop-in alternative to whestbench.dataset.create_dataset
for n_samples >= 10^8 scenarios where the flopscope CPU path is too slow.
Torch is an optional dependency: install via `pip install whestbench[gpu]`.
"""

from __future__ import annotations

import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import flopscope.numpy as fnp
import numpy as np

from .dataset import SCHEMA_VERSION, SEED_PROTOCOL_NAME, SEED_PROTOCOL_VERSION
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint


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
    flop_budget: int,
    seed: Optional[int] = None,
    output_path: "Path | str",
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    device: str = "auto",
    mlps_per_batch: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> Path:
    """Torch-backed analog of whestbench.dataset.create_dataset.

    Drop-in for create_dataset() at the same kwargs. Generates MLPs with the same
    seed protocol and produces an .npz with the same array layout. Output metadata
    self-identifies via backend="torch" and includes device/torch_version provenance.

    Statistical (not bitwise) equivalence with the flopscope CPU path holds at the
    same seed: per-neuron means agree within ~3e-5 at N=1e9 (MC noise).

    Args:
        n_mlps, n_samples, width, depth, flop_budget, seed, output_path, progress:
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
        Path to the written .npz.
    """
    torch = _require_torch()

    output_path = Path(output_path)
    resolved_device = _resolve_device(device)
    resolved_mlps_per_batch = (
        _auto_mlps_per_batch(n_mlps=n_mlps) if mlps_per_batch is None else int(mlps_per_batch)
    )
    resolved_chunk_size = (
        _auto_chunk_size(
            device=resolved_device, width=width, mlps_per_batch=resolved_mlps_per_batch
        )
        if chunk_size is None
        else int(chunk_size)
    )

    seed_sequence = (
        fnp.random.SeedSequence() if seed is None else fnp.random.SeedSequence(int(seed))
    )
    stream_seed = seed_sequence.spawn(3 * n_mlps)

    # Phase 1: generate MLPs on CPU (same protocol as create_dataset())
    mlps = []
    for i in range(n_mlps):
        weight_stream = fnp.random.default_rng(stream_seed[3 * i])
        estimator_seed_i = int(stream_seed[3 * i + 2].generate_state(1)[0])
        mlps.append(sample_mlp(width, depth, weight_stream, seed=estimator_seed_i))
        if progress is not None:
            progress({"phase": "generating", "completed": i + 1, "total": n_mlps})

    weights_array = np.stack([np.stack(mlp.weights) for mlp in mlps]).astype(np.float32)

    # Phase 2: sampling on device, batched across MLPs
    from ._simulation_torch import sample_layer_statistics_torch

    weights_device = torch.from_numpy(weights_array).to(resolved_device)
    chunks_per_mlp = math.ceil(n_samples / resolved_chunk_size)
    total_sampling_chunks = n_mlps * chunks_per_mlp

    all_means_list: List[np.ndarray] = []
    final_means_list: List[np.ndarray] = []
    avg_variances: List[float] = []
    sampling_budget_breakdowns: List[Dict[str, Any]] = []

    batch_starts = list(range(0, n_mlps, resolved_mlps_per_batch))
    for batch_start in batch_starts:
        batch_end = min(batch_start + resolved_mlps_per_batch, n_mlps)
        batch_size = batch_end - batch_start

        # Per-MLP torch generators seeded from the SeedSequence stream
        generators = []
        for i in range(batch_start, batch_end):
            torch_seed = int(stream_seed[3 * i + 1].generate_state(1)[0])
            gen = torch.Generator(device=resolved_device)
            gen.manual_seed(torch_seed)
            generators.append(gen)

        weights_slice = weights_device[batch_start:batch_end]

        def _on_chunk(
            event: Dict[str, Any],
            *,
            batch_start_local: int = batch_start,
            batch_size_local: int = batch_size,
        ) -> None:
            if progress is None:
                return
            local_completed = int(event.get("completed", 0))
            completed = batch_start_local * chunks_per_mlp + local_completed * batch_size_local
            progress(
                {
                    "phase": "sampling",
                    "completed": completed,
                    "total": total_sampling_chunks,
                    "mlp_index_range": (
                        batch_start_local + 1,
                        batch_start_local + batch_size_local,
                    ),
                    "n_mlps": n_mlps,
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
        avg_var_np = avg_var_batch.detach().to("cpu").numpy()

        for b in range(batch_size):
            all_means_list.append(layer_means_np[b])
            final_means_list.append(final_means_np[b])
            avg_variances.append(float(avg_var_np[b]))

    all_layer_means = np.stack(all_means_list).astype(np.float32)
    final_means = np.stack(final_means_list).astype(np.float32)

    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend": "torch",
        "seed_protocol": {
            "name": SEED_PROTOCOL_NAME,
            "version": SEED_PROTOCOL_VERSION,
            "seeded": seed is not None,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed_sequence.entropy),
        "n_mlps": n_mlps,
        "n_samples": n_samples,
        "width": width,
        "depth": depth,
        "flop_budget": flop_budget,
        "hardware": collect_hardware_fingerprint(),
        "torch_version": torch.__version__,
        "device": resolved_device,
        "mlps_per_batch": resolved_mlps_per_batch,
        "chunk_size": resolved_chunk_size,
    }
    if resolved_device == "cuda":
        metadata["cuda_device_name"] = torch.cuda.get_device_name()
        metadata["cuda_device_capability"] = list(torch.cuda.get_device_capability())
    elif resolved_device == "mps":
        metadata["mps_device_name"] = platform.processor() or "Apple Silicon"

    mlp_seeds = np.array([m.seed for m in mlps], dtype=np.int64)

    np.savez(
        output_path,
        metadata=np.array(json.dumps(metadata)),
        weights=weights_array,
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=np.array(avg_variances, dtype=np.float64),
        sampling_budget_breakdowns=np.array(json.dumps(sampling_budget_breakdowns)),
        mlp_seeds=mlp_seeds,
    )
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
      - matmul (n,w)@(w,w) per layer:  n * w^2 per layer
      - maximum (ReLU) per layer:      n * w per layer      (1 FLOP/element)
      - sum along axis=0 per layer:    n * w per layer      (input size, per flopscope docs)
      - power x_f64**2:               16 FLOPs/element * n * width
        (flopscope charges power at 16 FLOPs/element, unlike x*x which is 1)
      - sum for final_sum_sq:          n * width            (input size)

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
    matmul = depth * n_samples * width * width
    relu = depth * n_samples * width
    sum_layer = depth * n_samples * width
    power_sq = 16 * n_samples * width
    sum_sq = n_samples * width

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
