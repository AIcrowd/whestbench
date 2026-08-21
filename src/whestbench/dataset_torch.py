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
    _validate_config_name,
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
    config: str = "default",
    mlp_range: Optional[Tuple[int, int]] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    device: str = "auto",
    mlps_per_batch: Optional[int] = None,
    chunk_size: Optional[int] = None,
    compile: bool = False,
    flop_budget: Optional[int] = None,
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
        config: HF dataset config name for this split. Defaults to "default".
        device: "auto" | "cuda" | "mps" | "cpu". "auto" resolves cuda > mps > cpu.
            Explicit values error if unavailable (no silent CPU fallback).
            Note: bitwise reproducibility on CUDA additionally requires the
            caller to set torch.backends.cudnn.deterministic = True. This
            function does not set that flag — on CUDA, run-to-run output is
            deterministic in practice for the matmul/sum kernels used here,
            but not formally guaranteed by torch.
        flop_budget: The FLOP budget this bake is being run under, recorded per row in
            `sampling_budget_breakdown`. None (default) means the bake is NOT
            budget-limited -- the normal case for a ground-truth reference bake -- and
            records `flop_budget == flops_used` with `flops_remaining == 0`, so the
            identity holds without inventing a cap. This was previously hardcoded to
            1e15, a placeholder a production bake exceeds by ~34x while
            `max(0, budget - used)` clamped the remainder to 0, so every row claimed to
            have spent 34x its budget with none of it missing. It is unrelated to the
            estimator's FLOP budget at evaluation time.
        mlps_per_batch: How many MLPs to process in parallel on device.
            None (default) auto-tunes to min(n_mlps, 16).
        chunk_size: Samples per chunk on device. None (default) is memory-aware
            on cuda; fixed 65536 on mps/cpu.
        compile: When True (CUDA only), use an inductor-compiled + CUDA-graphed
            fused sampling kernel (~1.85x faster at width=256 on measured
            hardware). Default False. Bit-identical to the eager path on
            layer/final means; avg_variance may differ by ~1 fp64 ULP, which is
            within the parallel-bake contract's documented tolerance. Reproducible
            and parallel-merge bakes that enable compile must pin the torch
            version and re-bake any reference datasets — recorded in metadata as
            torch_compile=True for provenance.

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
    _validate_config_name(config)

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

    from .seeds import derive_seed_streams

    # Phase 1: generate MLPs on CPU (same protocol as create_dataset())
    mlps = []
    for slice_idx, i in enumerate(range(start, end)):
        weight_ss, _sample_ss, estimator_seed_i = derive_seed_streams(mlp_seeds[i])
        weight_stream = fnp.random.default_rng(weight_ss)
        mlps.append(sample_mlp(width, depth, weight_stream, seed=estimator_seed_i))
        if progress is not None:
            progress({"phase": "generating", "completed": slice_idx + 1, "total": end - start})

    # Names from ALL logical estimator seeds (so slice's names equal slice of single-host bake).
    # Mirrors create_dataset() so both backends produce identical name lists at same mlp_seeds.
    all_logical_seeds = [derive_seed_streams(mlp_seeds[i])[2] for i in range(n_mlps)]
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
        # Use logical index i to access sample_ss (spawn[1]) for the sample stream.
        generators = []
        for local_idx in range(batch_start_local, batch_end_local):
            i = local_idx + start  # logical index
            _weight_ss, sample_ss, _est = derive_seed_streams(mlp_seeds[i])
            torch_seed = int(sample_ss.generate_state(1)[0])
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
            compile=compile,
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
                    # The chunking the bake ACTUALLY ran with, not the CPU path's rule.
                    chunk_size=resolved_chunk_size,
                    wall_time_s=amortized_wall,
                    flop_budget=flop_budget,
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
        "split": split,
        "config": config,
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
        # torch.compile fused/graphed path engages on CUDA only; record what
        # actually ran so reproducible/parallel bakes can pin it (see
        # docs/how-to/parallel-bake.md § "Bit-equivalence requirements").
        "torch_compile": bool(compile) and resolved_device == "cuda",
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
    chunk_size: int,
    wall_time_s: float,
    flop_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Closed-form analog of flopscope's BudgetContext.summary_dict for the torch path.

    The torch path computes outside flopscope's instrumentation, so this helper synthesizes
    the same dict shape using analytical FLOP counts, per operation. Verified against
    flopscope's actual per-operation count by tests/test_torch_flop_synthesis.py.

    Output shape mirrors flopscope's normalized output exactly:
    - Top-level keys: flop_budget, flops_used, flops_remaining, wall_time_s,
      flopscope_backend_time_s, flopscope_overhead_time_s, residual_wall_time_s,
      by_namespace -- plus a "time_source": "bake" tag.
    - by_namespace is a FLAT dict keyed by dot-notation strings (e.g.
      "sampling.sample_layer_statistics"), NOT nested dicts.
    - Each namespace bucket carries `calls` and an `operations` map whose entries are
      {flop_cost, calls, flopscope_backend_time_s, flopscope_overhead_time_s} -- the same
      shape flopscope emits, and the same one whestbench.scoring._merge_operation_timing
      reads. Note `flop_cost`, not `flops`: the aggregator reads `flop_cost` and silently
      contributes zero for any other key.
    - Times are the bake machine's decomposition: the torch path runs outside flopscope, so
      backend = overhead = 0 and ALL bake wall clock is residual (wall = backend + overhead
      + residual holds). The "time_source" tag lets run-time reports attribute these to the
      bake machine instead of the current run -- informational, never billed (discourse
      #18093).

    CHUNK SIZE IS A PARAMETER, NOT A GUESS. It was previously re-derived here from
    `simulation._pick_chunk_size(width)`, which is the CPU path's rule and has nothing to do
    with what the torch path ran: `create_dataset_torch` resolves chunk_size from its own
    argument or from free VRAM. At width 1024 the two differ by 512x. That never showed up
    in `flops_used`, because the two chunk-dependent terms cancel --

        sum + add = 2(d+1)w(n - k) + 2(d+1)wk = 2(d+1)wn

    -- so k drops out of the TOTAL while surviving in the SPLIT. The total was right and
    every per-operation number derived from it would have been wrong, which is exactly the
    combination nobody notices.

    Formula derivation (matched against flopscope's operation-level accounting, from 0.10.0
    through current main; validated across single-chunk, multi-chunk and ragged-final-chunk
    dims by test_torch_flop_synthesis.py). Below, n = n_samples and k = n_chunks.

    flopscope bills under a dtype-aware model, so an op on float64 data costs 2x the same op
    on float32. sample_layer_statistics runs the forward pass in float32 (rate 1) and
    accumulates statistics in float64 (rate 2); the 2x is folded into the float64 terms
    below. Reductions cost (rows - 1) additions per output column, so a sum over a chunk of
    n_c rows costs (n_c - 1); summed over the k chunks that tile n_samples this is (n - k).
    Per-chunk whole-array accumulations happen once per chunk, so they scale with k.

    Forward pass -- float32 (rate 1), linear in n:
      - standard_normal((n, width)):   16 * n * width          (RNG: 16 FLOPs/element)
      - array wrap:                    n * width               (1 FLOP/element)
      - matmul (n,w)@(w,w) per layer:  depth * n * width * (2*width - 1)
        (BLAS MAC: width multiplies + (width-1) adds per output element)
      - maximum (ReLU) per layer:      depth * n * width       (1 FLOP/element)

    Statistics accumulation -- float64 (rate 2):
      - asarray float32->float64:      2 * width * (depth+1) * (n+1)
        ((depth+1) activation casts/chunk + 2 once-off casts; max(src,dst) rate = 2)
      - sum(axis=0) reductions:        2 * (depth+1) * (n - k) * width
      - add accumulations (per chunk): 2 * (depth+1) * k * width
      - power (x_f64**2):              32 * width * (n+1)      (16 FLOPs/element * rate 2)

    Buffer allocation -- free, but flopscope records the calls:
      - zeros:                         0 FLOPs, depth+1 calls  (layer_sums + final_sum_sq)

    Post-loop reductions -- float64 (rate 2), once; plus one float32 copy:
      - stack(layer_sums) -> (depth,w): 2 * depth * width
      - true_divide (/n_processed):    2 * width * (depth+1)   (layer means + final_sum_sq)
      - subtract (var = E[x^2] - mu^2): 2 * width
      - mean (avg variance):           2 * width
      - copy (final_mean):             width                   (float32, rate 1)

    Args:
        chunk_size: the chunk size the bake ACTUALLY ran with (create_dataset_torch passes
            its resolved value). Required -- there is no correct default.
        flop_budget: the budget this bake was run under, or None for an unbudgeted
            reference bake. None records flop_budget == flops_used and flops_remaining == 0,
            so `flop_budget - flops_used == flops_remaining` holds without inventing a cap.
    """
    w, d, n = width, depth, n_samples
    k = math.ceil(n / chunk_size)

    # A FINAL CHUNK OF EXACTLY ONE ROW IS BILLED DIFFERENTLY.
    # The reduction model charges (n_c - 1) accumulation steps per column, which is 0 for a
    # 1-row chunk. flopscope instead bills a fixed per-call cost there, making the closed
    # form exactly 2*(depth+1) low. Measured against live flopscope at five width/depth
    # combinations: the delta is independent of width and linear in depth, and appears only
    # for a final chunk of exactly one row (2, 3, 10, 904 and full-size finals all match to
    # the unit). Correcting it keeps flops_used exact for every chunking rather than for
    # most of them.
    degenerate_tail = 2 * (d + 1) if (k > 1 and n - chunk_size * (k - 1) == 1) else 0

    ops: Dict[str, tuple] = {
        # Forward pass -- float32 (rate 1), once per chunk.
        "random.Generator.standard_normal": (16 * n * w, k),
        "array": (n * w, k),
        "matmul": (d * n * w * (2 * w - 1), d * k),
        "maximum": (d * n * w, d * k),
        # Statistics accumulation -- float64 (rate 2), (depth+1) per chunk.
        "asarray": (2 * w * (d + 1) * (n + 1), (d + 1) * k + 2),
        "sum": (2 * (d + 1) * (n - k) * w + degenerate_tail, (d + 1) * k),
        "add": (2 * (d + 1) * k * w, (d + 1) * k),
        "power": (32 * w * (n + 1), k + 1),
        # Free, but counted.
        "zeros": (0, d + 1),
        # Post-loop reductions -- once.
        "stack": (2 * d * w, 1),
        "true_divide": (2 * w * (d + 1), 2),
        "subtract": (2 * w, 1),
        "mean": (2 * w, 1),
        "copy": (w, 1),
    }

    total = sum(flops for flops, _ in ops.values())
    calls = sum(c for _, c in ops.values())

    # An unbudgeted reference bake records budget == used, so the identity holds without
    # fabricating a cap. A stated budget is recorded verbatim, and the remainder is the
    # true one: clamping a negative remainder to 0 reports "exactly exhausted" for a run
    # that blew through its budget, which is the more dangerous of the two readings.
    budget = total if flop_budget is None else int(flop_budget)
    flops_remaining = budget - total

    return {
        "flop_budget": budget,
        "flops_used": total,
        "flops_remaining": flops_remaining,
        "wall_time_s": wall_time_s,
        "flopscope_backend_time_s": 0.0,
        "flopscope_overhead_time_s": 0.0,
        "residual_wall_time_s": wall_time_s,
        "time_source": "bake",
        "by_namespace": {
            "sampling.sample_layer_statistics": {
                "flops_used": total,
                "calls": calls,
                "flopscope_backend_time_s": 0.0,
                "flopscope_overhead_time_s": 0.0,
                "operations": {
                    name: {
                        "flop_cost": flops,
                        "calls": c,
                        "flopscope_backend_time_s": 0.0,
                        "flopscope_overhead_time_s": 0.0,
                    }
                    for name, (flops, c) in ops.items()
                },
            }
        },
        # Extra keys are safe: scoring._normalize_sampling_budget_breakdown and
        # _aggregate_budget_breakdowns both build a fresh dict from named keys.
        #
        # chunk_size is the point of this block. Float64 accumulation is not associative,
        # so reproducing a bake's means bit-for-bit requires the same chunk decomposition —
        # and chunk_size is not recorded anywhere else, which is why the card's re-bake
        # recipe could only promise the same MLPs and statistically equivalent means.
        # Recording it per row closes that gap for the value that auto-resolves from free
        # VRAM and is otherwise unrecoverable after the fact.
        "provenance": {
            "method": "closed-form",
            "operation_model": "whestbench.simulation.sample_layer_statistics",
            "chunk_size": chunk_size,
            "n_chunks": k,
            "note": (
                "The torch backend computes outside flopscope's instrumentation, so these "
                "are closed-form FLOP counts for the numpy reference implementation's "
                "operations, evaluated at the chunking this bake actually used. Pass "
                "chunk_size back to reproduce the same accumulation order."
            ),
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
