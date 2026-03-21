# src/network_estimation/profiler.py
"""Profiling engine for simulation backends.

Benchmarks every available simulation backend (numpy, pytorch, numba, jax,
scipy, cython) head-to-head on the two core operations — ``run_mlp`` and
``sample_layer_statistics`` — across a configurable grid of network sizes and sample
counts.

Workflow
--------
1. **Discovery** — detect which backends are installed.
2. **Correctness check** — each backend's outputs are compared to the NumPy
   reference implementation before any timing is recorded.  Backends that
   fail are excluded from the timing sweep.
3. **Timing sweep** — for every (backend, operation, width, depth, n_samples)
   combination the operation is timed ``n_iterations`` times (default 3).
   Garbage collection is disabled during measurement.  Results are expressed
   as median wall-clock seconds and as a speedup factor relative to NumPy.
4. **Output** — a Rich-formatted terminal table and, optionally, a
   machine-readable JSON file that includes hardware metadata and library
   versions for reproducibility.

CLI usage
---------
::

    nestim profile-simulation                        # standard sweep, all backends
    nestim profile-simulation --preset quick          # fast smoke test
    nestim profile-simulation --preset exhaustive     # full matrix
    nestim profile-simulation --backends numpy,pytorch
    nestim profile-simulation --output results.json   # save JSON report

Presets
-------
``super-quick``
    One width (64), one depth (4), 1 000 samples — sub-second, for testing
    the debug loop.
``quick``
    One width (256), two depths, two sample counts — finishes in seconds.
``standard`` *(default)*
    Two widths, five depths, three sample counts — a few minutes.
``exhaustive``
    Three widths, five depths, five sample counts (up to 16.7 M) — thorough
    but slow.

See Also
--------
``nestim profile-simulation --help`` for the full list of CLI flags.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .domain import MLP
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint
from .simulation import (
    sample_layer_statistics as ref_sample_layer_statistics,
    run_mlp as ref_run_mlp,
)
from .simulation_backend import PrimitiveBreakdown, SimulationBackend
from .simulation_backends import ALL_BACKEND_NAMES, INSTALL_HINTS, get_available_backends


@dataclass
class PresetConfig:
    """Parameter sweep grid for profiling.

    Each preset defines a cross-product of network shapes and sample counts.
    Every (width, depth, n_samples) triple is evaluated for both ``run_mlp``
    and ``sample_layer_statistics``.

    Attributes:
        widths: Hidden-layer widths to benchmark (e.g. ``[64, 256]``).
        depths: Network depths (number of layers) to benchmark.
        n_samples_list: Number of random input samples per timing call.
    """

    widths: List[int]
    depths: List[int]
    n_samples_list: List[int]


PRESETS: Dict[str, PresetConfig] = {
    "super-quick": PresetConfig(
        widths=[64],
        depths=[4],
        n_samples_list=[1_000],
    ),
    "quick": PresetConfig(
        widths=[256],
        depths=[4, 32],
        n_samples_list=[10_000, 100_000],
    ),
    "standard": PresetConfig(
        widths=[64, 256],
        depths=[4, 16, 32, 64, 128],
        n_samples_list=[10_000, 100_000, 1_000_000],
    ),
    "exhaustive": PresetConfig(
        widths=[64, 128, 256],
        depths=[4, 16, 32, 64, 128],
        n_samples_list=[10_000, 100_000, 500_000, 1_000_000, 16_700_000],
    ),
}


def format_dims(width: int, depth: int, n_samples: int) -> str:
    """Format dimensions as compact 'w×d×n' string with k/M suffixes."""
    if n_samples >= 1_000_000:
        if n_samples % 1_000_000 == 0:
            n_str = f"{n_samples // 1_000_000}M"
        else:
            n_str = f"{n_samples / 1_000_000:.1f}M"
    elif n_samples >= 1_000:
        if n_samples % 1_000 == 0:
            n_str = f"{n_samples // 1_000}k"
        else:
            n_str = f"{n_samples / 1_000:.1f}k"
    else:
        n_str = str(n_samples)
    return f"{width}×{depth}×{n_str}"


@dataclass
class CorrectnessResult:
    """Result of a pre-flight correctness check for one backend.

    Attributes:
        backend_name: Identifier of the backend that was tested.
        passed: ``True`` if the backend matched the NumPy reference.
        error: Human-readable error message when ``passed`` is ``False``.
    """

    backend_name: str
    passed: bool
    error: str = ""


@dataclass
class TimingResult:
    """Timing measurement for one (backend, operation, parameter) combination.

    Attributes:
        backend_name: Identifier of the backend.
        operation: ``"run_mlp"`` or ``"sample_layer_statistics"``.
        width: Hidden-layer width of the network used.
        depth: Depth (number of layers) of the network used.
        n_samples: Number of random input samples.
        times: Raw wall-clock seconds for each iteration.
        median_time: Median of *times* (seconds).
        speedup_vs_numpy: ``numpy_median / median_time``.
            Values > 1 mean the backend is faster than NumPy.
    """

    backend_name: str
    operation: str
    width: int
    depth: int
    n_samples: int
    times: List[float]
    median_time: float
    speedup_vs_numpy: float
    warmup_time: Optional[float] = None
    breakdown: Optional[PrimitiveBreakdown] = None
    error: str = ""


def _collect_hardware_info(max_threads: Optional[int] = None) -> Dict[str, Any]:
    """Collect hardware info for the profiling report.

    Delegates to :func:`collect_hardware_fingerprint` for detailed CPU,
    RAM, and platform info, then adds profiler-specific fields like
    ``max_threads``.
    """
    info = collect_hardware_fingerprint()
    if max_threads is not None:
        info["max_threads"] = max_threads
    return info


def _collect_backend_versions(backend_names: List[str]) -> Dict[str, str]:
    """Collect version strings for each backend's underlying library."""
    versions: Dict[str, str] = {}
    versions["numpy"] = np.__version__
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        pass
    if "pytorch" in backend_names:
        try:
            import torch
            versions["pytorch"] = torch.__version__
        except ImportError:
            pass
    if "numba" in backend_names:
        try:
            import numba
            versions["numba"] = numba.__version__
        except ImportError:
            pass
    if "jax" in backend_names:
        try:
            import jax
            versions["jax"] = jax.__version__
        except ImportError:
            pass
    if "cython" in backend_names:
        try:
            import Cython
            versions["cython"] = Cython.__version__
        except ImportError:
            pass
    return versions


def correctness_check(
    backend: SimulationBackend,
) -> CorrectnessResult:
    """Pre-flight correctness check against the NumPy reference implementation.

    Validates both ``run_mlp`` (exact match, rtol=1e-5, atol=1e-6) and
    ``sample_layer_statistics`` (statistical match, atol=0.15 for means, relative
    tolerance for variance).  A backend that fails this check is excluded
    from subsequent timing sweeps.

    Args:
        backend: The simulation backend instance to verify.

    Returns:
        A :class:`CorrectnessResult` indicating pass/fail and any error
        details.
    """
    try:
        mlp = sample_mlp(8, 4, np.random.default_rng(42))
        inputs = np.random.default_rng(123).standard_normal((64, 8)).astype(np.float32)

        # Exact match for run_mlp
        ref = ref_run_mlp(mlp, inputs)
        result = backend.run_mlp(mlp, inputs)
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-6)

        # Statistical match for sample_layer_statistics
        ref_means, ref_final, ref_var = ref_sample_layer_statistics(mlp, 1000)
        fast_means, fast_final, fast_var = backend.sample_layer_statistics(mlp, 1000)
        np.testing.assert_allclose(fast_means, ref_means, atol=0.15)
        np.testing.assert_allclose(fast_final, ref_final, atol=0.15)
        # Variance can differ more with only 1000 samples
        if abs(ref_var) > 1e-6:
            assert abs(fast_var - ref_var) < max(0.5 * abs(ref_var), 0.1)

        return CorrectnessResult(backend_name=backend.name, passed=True)

    except Exception as e:
        return CorrectnessResult(
            backend_name=backend.name, passed=False, error=str(e)
        )


def _time_run_mlp(
    backend: SimulationBackend, mlp: MLP, n_samples: int
) -> float:
    """Time a single run_mlp call."""
    inputs = np.random.randn(n_samples, mlp.width).astype(np.float32)
    t0 = time.perf_counter()
    backend.run_mlp(mlp, inputs)
    return time.perf_counter() - t0


def _time_sample_layer_statistics(
    backend: SimulationBackend, mlp: MLP, n_samples: int
) -> float:
    """Time a single sample_layer_statistics call."""
    t0 = time.perf_counter()
    backend.sample_layer_statistics(mlp, n_samples)
    return time.perf_counter() - t0


def _time_run_mlp_matmul_only(
    backend: SimulationBackend, mlp: MLP, n_samples: int
) -> float:
    """Time a single run_mlp_matmul_only call."""
    inputs = np.random.randn(n_samples, mlp.width).astype(np.float32)
    t0 = time.perf_counter()
    backend.run_mlp_matmul_only(mlp, inputs)
    return time.perf_counter() - t0


def run_timing_sweep(
    backends: Dict[str, SimulationBackend],
    preset: PresetConfig,
    n_iterations: int = 3,
    progress_callback: Optional[Any] = None,
) -> Tuple[List[TimingResult], Dict[str, List[float]]]:
    """Run the full timing sweep across all backends and parameter combinations.

    For each (width, depth, n_samples, operation) tuple the sweep:

    1. Performs an untimed warmup call (critical for JIT backends like Numba
       and JAX).
    2. Disables garbage collection and records *n_iterations* timed calls.
    3. Computes the median time and speedup relative to the NumPy baseline.

    Args:
        backends: Mapping of backend name to instantiated backend.  Should
            only contain backends that already passed the correctness check.
        preset: The parameter grid to iterate over.
        n_iterations: Number of timed repetitions per combination (default 3).
        progress_callback: Optional callable invoked after each measurement.
            Receives keyword arguments ``backend_name``, ``operation``,
            ``width``, ``depth``, and ``n_samples``.

    Returns:
        A tuple of ``(results, numpy_baselines)`` where *results* is a list
        of :class:`TimingResult` objects and *numpy_baselines* maps
        ``"op:width:depth:n_samples"`` keys to the raw NumPy timing lists.
    """
    results: List[TimingResult] = []
    numpy_baselines: Dict[str, List[float]] = {}

    operations = ["run_mlp", "run_mlp_matmul_only", "sample_layer_statistics"]
    time_fns = {
        "run_mlp": _time_run_mlp,
        "run_mlp_matmul_only": _time_run_mlp_matmul_only,
        "sample_layer_statistics": _time_sample_layer_statistics,
    }

    for width in preset.widths:
        for depth in preset.depths:
            mlp = sample_mlp(width, depth, np.random.default_rng(42))
            for n_samples in preset.n_samples_list:
                for op in operations:
                    key = f"{op}:{width}:{depth}:{n_samples}"
                    time_fn = time_fns[op]

                    for backend_name, backend in backends.items():
                        try:
                            # Timed warmup (critical for JIT backends)
                            warmup_t0 = time.perf_counter()
                            time_fn(backend, mlp, min(n_samples, 1000))
                            warmup_time = time.perf_counter() - warmup_t0

                            gc_was_enabled = gc.isenabled()
                            gc.disable()
                            times = []
                            try:
                                for _ in range(n_iterations):
                                    t = time_fn(backend, mlp, n_samples)
                                    times.append(t)
                            finally:
                                if gc_was_enabled:
                                    gc.enable()

                            median_t = float(np.median(times))

                            # Store numpy baselines
                            if backend_name == "numpy":
                                numpy_baselines[key] = times

                            # Compute speedup vs numpy
                            numpy_times = numpy_baselines.get(key)
                            if numpy_times is not None:
                                numpy_median = float(np.median(numpy_times))
                                speedup = numpy_median / median_t if median_t > 0 else float("inf")
                            else:
                                speedup = 1.0

                            results.append(TimingResult(
                                backend_name=backend_name,
                                operation=op,
                                width=width,
                                depth=depth,
                                n_samples=n_samples,
                                times=times,
                                median_time=median_t,
                                speedup_vs_numpy=speedup,
                                warmup_time=warmup_time,
                            ))

                        except (MemoryError, Exception) as exc:
                            # Log and skip — partial results are better than none.
                            err_msg = f"{type(exc).__name__}: {exc}"
                            print(
                                f"[warning] {backend_name} {op} "
                                f"w={width} d={depth} n={n_samples:,} "
                                f"skipped: {err_msg}",
                                flush=True,
                            )
                            results.append(TimingResult(
                                backend_name=backend_name,
                                operation=op,
                                width=width,
                                depth=depth,
                                n_samples=n_samples,
                                times=[],
                                median_time=-1.0,
                                speedup_vs_numpy=0.0,
                                error=err_msg,
                            ))

                        if progress_callback:
                            progress_callback(
                                backend_name=backend_name,
                                operation=op,
                                width=width,
                                depth=depth,
                                n_samples=n_samples,
                            )

    # Post-process: compute PrimitiveBreakdown by subtraction
    # matmul_time = run_mlp_matmul_only, relu_time = run_mlp - matmul_only
    matmul_only_lookup: Dict[Tuple[str, int, int, int], float] = {}
    for tr in results:
        if tr.operation == "run_mlp_matmul_only" and not tr.error:
            matmul_only_lookup[(tr.backend_name, tr.width, tr.depth, tr.n_samples)] = tr.median_time

    for tr in results:
        if tr.operation == "run_mlp" and not tr.error:
            key = (tr.backend_name, tr.width, tr.depth, tr.n_samples)
            matmul_t = matmul_only_lookup.get(key)
            if matmul_t is not None:
                relu_t = max(0.0, tr.median_time - matmul_t)
                tr.breakdown = PrimitiveBreakdown(
                    matmul_total=matmul_t,
                    relu_total=relu_t,
                    fused_total=tr.median_time,
                )

    return results, numpy_baselines


def format_verbose_output(
    correctness_results: List[CorrectnessResult],
    timing_results: List[TimingResult],
    skipped_backends: Dict[str, str],
    hardware_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Format profiling results as a Rich-rendered terminal table.

    The output includes three sections:

    * **Skipped backends** — lists backends that were not installed, with
      ``pip install`` hints.
    * **Correctness check** — pass/fail status for each backend.
    * **Timing table** — median time, speedup vs NumPy (green for faster,
      red for slower), and correctness status per combination.

    Returns:
        A string containing ANSI-styled output suitable for printing to a
        terminal.
    """
    from rich.console import Console
    from rich.table import Table
    import io

    # Filter out errored timing entries for display
    timing_results = [tr for tr in timing_results if not tr.error]

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=140)

    # Hardware summary
    if hardware_info:
        console.print("\n[bold]Hardware[/bold]")
        _hw_label = {
            "platform": "Platform",
            "machine": "Architecture",
            "cpu_brand": "CPU",
            "cpu_count_physical": "Physical Cores",
            "cpu_count_logical": "Logical Cores",
            "ram_total_bytes": "RAM",
            "python_version": "Python",
            "numpy_version": "NumPy",
            "max_threads": "Thread Limit",
        }
        for key, label in _hw_label.items():
            val = hardware_info.get(key)
            if val is None:
                continue
            if key == "ram_total_bytes":
                val = f"{val / (1024**3):.1f} GB"
            console.print(f"  {label}: {val}")

    # Skipped backends
    if skipped_backends:
        console.print("\n[bold yellow]Skipped backends:[/bold yellow]")
        for name, hint in skipped_backends.items():
            console.print(f"  {name}: not installed. Install: {hint}")

    # Correctness results
    console.print("\n[bold]Pre-flight Correctness Check[/bold]")
    for cr in correctness_results:
        status = "[green]PASS[/green]" if cr.passed else f"[red]FAIL: {cr.error}[/red]"
        console.print(f"  {cr.backend_name}: {status}")

    # Build a set of passed backend names for status lookup
    passed_names = {cr.backend_name for cr in correctness_results if cr.passed}

    # Timing table (run_mlp + sample_layer_statistics only, hide matmul_only internal op)
    non_profiled = [tr for tr in timing_results if tr.operation in ("run_mlp", "sample_layer_statistics")]
    if non_profiled:
        table = Table(title="\nTiming Results", show_lines=True)
        table.add_column("Backend", style="cyan")
        table.add_column("Operation")
        table.add_column("Width", justify="right")
        table.add_column("Depth", justify="right")
        table.add_column("N_Samples", justify="right")
        table.add_column("Median Time (s)", justify="right")
        table.add_column("Speedup vs NumPy", justify="right")
        table.add_column("Status")

        for tr in non_profiled:
            speedup_str = f"{tr.speedup_vs_numpy:.2f}x"
            if tr.speedup_vs_numpy > 1.0:
                speedup_str = f"[green]{speedup_str}[/green]"
            elif tr.speedup_vs_numpy < 1.0:
                speedup_str = f"[red]{speedup_str}[/red]"

            status = "[green]OK[/green]" if tr.backend_name in passed_names else "[red]FAIL[/red]"

            table.add_row(
                tr.backend_name,
                tr.operation,
                str(tr.width),
                str(tr.depth),
                f"{tr.n_samples:,}",
                f"{tr.median_time:.4f}",
                speedup_str,
                status,
            )

        console.print(table)

    # Matmul vs ReLU breakdown (derived by subtraction: fused - matmul_only)
    with_breakdown = [tr for tr in timing_results if tr.breakdown is not None]
    if with_breakdown:
        bd_table = Table(title="\nMatmul vs ReLU Breakdown (by subtraction)", show_lines=True)
        bd_table.add_column("Backend", style="cyan")
        bd_table.add_column("Width", justify="right")
        bd_table.add_column("Depth", justify="right")
        bd_table.add_column("N_Samples", justify="right")
        bd_table.add_column("Fused (s)", justify="right")
        bd_table.add_column("Matmul (s)", justify="right")
        bd_table.add_column("Matmul %", justify="right")
        bd_table.add_column("ReLU (s)", justify="right")
        bd_table.add_column("ReLU %", justify="right")

        for tr in with_breakdown:
            bd = tr.breakdown
            bd_table.add_row(
                tr.backend_name,
                str(tr.width),
                str(tr.depth),
                f"{tr.n_samples:,}",
                f"{bd.fused_total:.4f}",
                f"{bd.matmul_total:.4f}",
                f"[bold]{bd.matmul_pct:.1f}%[/bold]",
                f"{bd.relu_total:.6f}",
                f"{bd.relu_pct:.1f}%",
            )

        console.print(bd_table)

    return buf.getvalue()


def format_compact_output(
    correctness_results: List[CorrectnessResult],
    timing_results: List[TimingResult],
    skipped_backends: Dict[str, str],
    hardware_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Format profiling results as a compact Rich-rendered summary.

    Produces a 3-zone view: one-line hardware context, a leaderboard showing
    which backend wins, and a single detail table merging timing and primitive
    breakdown data.

    Args:
        correctness_results: Per-backend correctness check results.
        timing_results: All timing measurements from the sweep.
        skipped_backends: Mapping of skipped backend names to install hints.
        hardware_info: Optional hardware fingerprint dictionary.

    Returns:
        A string containing ANSI-styled output suitable for printing to a
        terminal.
    """
    from collections import defaultdict
    from rich.console import Console
    from rich.table import Table
    import io

    term_width = shutil.get_terminal_size((120, 24)).columns
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=term_width)

    # Filter out errored timing entries (e.g. OOM) for display purposes
    timing_results = [tr for tr in timing_results if not tr.error]

    passed_names = {cr.backend_name for cr in correctness_results if cr.passed}
    all_failed = len(passed_names) == 0

    # --- Zone 1: Context lines ---

    # Line 1 — Hardware
    if hardware_info is not None:
        parts: List[str] = []
        os_name = hardware_info.get("os", "")
        machine = hardware_info.get("machine", "")
        if os_name or machine:
            parts.append(f"{os_name} {machine}".strip())

        cores = hardware_info.get("cpu_count_physical")
        if cores is None:
            cores = hardware_info.get("cpu_count_logical")
        if cores is not None:
            parts.append(f"{cores} cores")

        ram_bytes = hardware_info.get("ram_total_bytes")
        if ram_bytes is not None:
            parts.append(f"{ram_bytes / (1024**3):.1f} GB")

        py_ver = hardware_info.get("python_version")
        if py_ver is not None:
            parts.append(f"Python {py_ver}")

        np_ver = hardware_info.get("numpy_version")
        if np_ver is not None:
            parts.append(f"NumPy {np_ver}")

        if parts:
            console.print(" \u00b7 ".join(parts), highlight=False)

    # Line 2 — Skipped
    if skipped_backends:
        names = ", ".join(skipped_backends.keys())
        console.print(f"Skipped: {names} (--backends-help for install info)")

    # Line 3 — Correctness
    if all_failed:
        console.print(
            "No backends passed correctness checks. Use --verbose for error details."
        )
        return buf.getvalue()

    corr_parts: List[str] = []
    for cr in correctness_results:
        if cr.passed:
            corr_parts.append(f"{cr.backend_name} \u2713")
        else:
            corr_parts.append(
                f"\u26a0 FAIL: {cr.backend_name} (use --verbose for details)"
            )
    console.print("Correctness: " + "  ".join(corr_parts))

    # --- Helpers for grouping ---
    def _dim_key(tr: TimingResult) -> Tuple[int, int, int]:
        return (tr.width, tr.depth, tr.n_samples)

    def _dim_label(w: int, d: int, n: int) -> str:
        return f"w={w} d={d} n={n:,}"

    # Filter to passed backends only
    passed_timing = [tr for tr in timing_results if tr.backend_name in passed_names]

    # --- Zone 2: Leaderboard (only if 2+ backends passed) ---
    if len(passed_names) >= 2:
        run_mlp_results = [tr for tr in passed_timing if tr.operation == "run_mlp"]

        # Group by dims
        groups: Dict[Tuple[int, int, int], List[TimingResult]] = defaultdict(list)
        for tr in run_mlp_results:
            groups[_dim_key(tr)].append(tr)

        console.print()
        console.rule("Leaderboard")

        multiple_groups = len(groups) > 1

        for dim_key in sorted(groups.keys()):
            entries = sorted(groups[dim_key], key=lambda t: t.median_time)
            if multiple_groups:
                console.print(f"  [bold]{_dim_label(*dim_key)}[/bold]")

            # Determine baseline backend (numpy if present, else fastest)
            baseline_name = "numpy" if any(
                e.backend_name == "numpy" for e in entries
            ) else entries[0].backend_name

            baseline_median = next(
                e.median_time for e in entries if e.backend_name == baseline_name
            )

            for rank, tr in enumerate(entries, 1):
                speedup_str = ""
                if tr.backend_name != baseline_name:
                    sp = baseline_median / tr.median_time if tr.median_time > 0 else float("inf")
                    if sp > 1.0:
                        speedup_str = f"  [green]({sp:.2f}x)[/green]"
                    elif sp < 1.0:
                        speedup_str = f"  [red]({sp:.2f}x)[/red]"
                    else:
                        speedup_str = f"  ({sp:.2f}x)"
                console.print(
                    f"  #{rank}  {tr.backend_name:<12} {tr.median_time:.4f}s{speedup_str}",
                    highlight=False,
                )

    # --- Zone 3: Compact Detail Table ---
    console.print()
    console.rule("Detail")

    table = Table(show_header=True, show_lines=False, pad_edge=False)
    table.add_column("Backend", style="cyan")
    table.add_column("Dims")
    table.add_column("run_mlp", justify="right")
    table.add_column("sample_layer_statistics", justify="right")
    table.add_column("Matmul", justify="right")
    table.add_column("ReLU", justify="right")
    table.add_column("", justify="center")  # status checkmark

    # Build lookup for breakdown data: (backend, dims) -> PrimitiveBreakdown
    breakdown_lookup: Dict[Tuple[str, Tuple[int, int, int]], Any] = {}
    for tr in timing_results:
        if tr.operation == "run_mlp" and tr.breakdown is not None:
            breakdown_lookup[(tr.backend_name, _dim_key(tr))] = tr.breakdown

    # Build lookup for timing: (backend, dims, operation) -> median_time
    time_lookup: Dict[Tuple[str, Tuple[int, int, int], str], float] = {}
    for tr in passed_timing:
        if tr.operation in ("run_mlp", "sample_layer_statistics"):
            time_lookup[(tr.backend_name, _dim_key(tr), tr.operation)] = tr.median_time

    # Collect all dim groups present in timing
    all_dims: List[Tuple[int, int, int]] = []
    seen_dims: set = set()
    for tr in passed_timing:
        dk = _dim_key(tr)
        if dk not in seen_dims:
            seen_dims.add(dk)
            all_dims.append(dk)
    all_dims.sort()

    # For each dim group, sort backends by run_mlp time (fastest first)
    first_group = True
    for dim_key in all_dims:
        # Get backends that have run_mlp results for this dim
        backend_times: List[Tuple[str, float]] = []
        for bname in passed_names:
            t = time_lookup.get((bname, dim_key, "run_mlp"))
            if t is not None:
                backend_times.append((bname, t))
        backend_times.sort(key=lambda x: x[1])

        if not first_group:
            table.add_section()
        first_group = False

        for bname, _ in backend_times:
            run_mlp_t = time_lookup.get((bname, dim_key, "run_mlp"))
            out_stats_t = time_lookup.get((bname, dim_key, "sample_layer_statistics"))
            bd = breakdown_lookup.get((bname, dim_key))

            run_mlp_str = f"{run_mlp_t:.4f}s" if run_mlp_t is not None else "\u2014"
            out_stats_str = f"{out_stats_t:.4f}s" if out_stats_t is not None else "\u2014"

            def _bar(pct: float, color: str) -> str:
                n_blocks = int(pct / 20)
                n_blocks = min(n_blocks, 5)
                bar = "\u2588" * n_blocks
                if bar:
                    return f"[{color}]{bar}[/{color}] {pct:.0f}%"
                return f"{pct:.0f}%"

            if bd is not None:
                matmul_str = _bar(bd.matmul_pct, "blue")
                relu_str = _bar(bd.relu_pct, "green")
            else:
                matmul_str = "\u2014"
                relu_str = "\u2014"

            status = "\u2713" if bname in passed_names else "\u2717"

            table.add_row(
                bname,
                format_dims(dim_key[0], dim_key[1], dim_key[2]),
                run_mlp_str,
                out_stats_str,
                matmul_str,
                relu_str,
                status,
            )

    console.print(table)

    # Footer
    console.print("  [blue]\u2588[/blue] Matmul  [green]\u2588[/green] ReLU (by subtraction)", highlight=False)
    console.print()
    console.print(
        "[dim italic]Use --verbose for full timing tables with raw times[/dim italic]"
    )

    return buf.getvalue()


def format_json_output(
    correctness_results: List[CorrectnessResult],
    timing_results: List[TimingResult],
    skipped_backends: Dict[str, str],
    backend_names: Optional[List[str]] = None,
    hardware_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Format profiling results as a JSON-serializable dictionary.

    The top-level keys are:

    * ``hardware`` — platform, CPU, Python version (for reproducibility).
    * ``backend_versions`` — version strings for each library.
    * ``skipped_backends`` — backends not installed, with install hints.
    * ``correctness`` — per-backend pass/fail + error message.
    * ``timing`` — per-combination raw times, median, and speedup.

    Use ``--output results.json`` on the CLI to write this automatically.
    """
    return {
        "hardware": hardware_info or _collect_hardware_info(),
        "backend_versions": _collect_backend_versions(backend_names or []),
        "skipped_backends": skipped_backends,
        "correctness": [
            {"backend": cr.backend_name, "passed": cr.passed, "error": cr.error}
            for cr in correctness_results
        ],
        "timing": [
            {
                "backend": tr.backend_name,
                "operation": tr.operation,
                "width": tr.width,
                "depth": tr.depth,
                "n_samples": tr.n_samples,
                "times": tr.times,
                "median_time": tr.median_time,
                "speedup_vs_numpy": tr.speedup_vs_numpy,
                **({"warmup_time": tr.warmup_time} if tr.warmup_time is not None else {}),
                **({"breakdown": tr.breakdown.to_dict()} if tr.breakdown else {}),
                **({"error": tr.error} if tr.error else {}),
            }
            for tr in timing_results
        ],
    }


def run_profile(
    preset_name: str = "standard",
    backend_filter: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show_progress: bool = False,
    max_threads: Optional[int] = None,
    verbose: bool = False,
    log_progress: bool = False,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Run the complete profiling pipeline and return formatted results.

    This is the main entry point used by ``nestim profile-simulation``.
    It discovers backends, runs correctness checks, performs the timing
    sweep, and formats the output.

    Args:
        preset_name: One of ``"quick"``, ``"standard"``, or ``"exhaustive"``.
            Controls the size of the parameter grid.  See module docstring
            for details.
        backend_filter: If provided, only these backends are profiled.
            Names must be from :data:`ALL_BACKEND_NAMES`; a ``ValueError``
            is raised for unknown names.
        output_path: Optional file path to write a JSON report.  The file
            includes hardware info and library versions for reproducibility.
        show_progress: When ``True``, display a Rich progress bar in the
            terminal during correctness checks and timing sweeps.
        max_threads: If set, cap all backends to at most this many CPU
            threads.  Affects BLAS (OpenBLAS/MKL), Numba, PyTorch, and
            JAX/XLA thread pools.
        log_progress: When ``True``, print one line per benchmark step to
            stdout.  Designed for non-TTY environments (e.g. containers)
            where the Rich progress bar is invisible.

    Returns:
        A tuple of ``(terminal_output, json_data)`` where *terminal_output*
        is a Rich-formatted string and *json_data* is the JSON dict (or
        ``None`` if *output_path* was not set).

    Raises:
        ValueError: If *backend_filter* contains an unrecognised backend
            name.
        KeyError: If *preset_name* is not a valid preset.

    Example::

        terminal_output, json_data = run_profile(
            preset_name="quick",
            backend_filter=["numpy", "pytorch"],
            output_path="results.json",
            show_progress=True,
        )
        print(terminal_output)
    """
    if max_threads is not None:
        from .concurrency import apply_thread_limit
        apply_thread_limit(max_threads)

    # Collect hardware info early (before timing, so it doesn't interfere)
    hardware_info = _collect_hardware_info(max_threads=max_threads)

    # Suppress float32 overflow warnings from deep random-weight networks
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

    preset = PRESETS[preset_name]

    # Discover backends
    available = get_available_backends()
    if backend_filter:
        for name in backend_filter:
            if name not in ALL_BACKEND_NAMES:
                raise ValueError(
                    f"Unknown backend: {name!r}. Valid backends: {list(ALL_BACKEND_NAMES)}"
                )
        available = {k: v for k, v in available.items() if k in backend_filter}

    # Track skipped backends
    skipped: Dict[str, str] = {}
    all_names = list(backend_filter) if backend_filter else list(ALL_BACKEND_NAMES)
    for name in all_names:
        if name not in available:
            skipped[name] = INSTALL_HINTS.get(name, "")

    # Instantiate backends
    backend_instances: Dict[str, SimulationBackend] = {}
    for name, cls in available.items():
        backend_instances[name] = cls()

    # Set up progress display
    use_rich_progress = show_progress and not log_progress
    progress_ctx: Any = None
    if use_rich_progress:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )
            from rich.table import Column

            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn(
                    "[progress.description]{task.description}",
                    table_column=Column(min_width=60),
                ),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
        except ImportError:
            pass

    # Pre-flight correctness check
    correctness_results: List[CorrectnessResult] = []
    passed_backends: Dict[str, SimulationBackend] = {}

    if progress_ctx is not None:
        correctness_task = progress_ctx.add_task(
            "Correctness checks", total=len(backend_instances)
        )
        progress_ctx.start()

    for name, backend in backend_instances.items():
        if progress_ctx is not None:
            progress_ctx.update(
                correctness_task,
                description=f"Correctness check [cyan]{name:<8}[/]",
            )
        if log_progress:
            print(f"[correctness] {name} ...", end=" ", flush=True)
        cr = correctness_check(backend)
        correctness_results.append(cr)
        if cr.passed:
            passed_backends[name] = backend
        if log_progress:
            print("PASS" if cr.passed else f"FAIL: {cr.error}", flush=True)
        if progress_ctx is not None:
            progress_ctx.advance(correctness_task)

    # Timing sweep (only on backends that passed correctness)
    timing_results: List[TimingResult] = []
    timing_task = None
    if passed_backends:
        n_combos = (
            len(preset.widths)
            * len(preset.depths)
            * len(preset.n_samples_list)
            * 3  # run_mlp + run_mlp_matmul_only + sample_layer_statistics
            * len(passed_backends)
        )

        callback: Any = None

        if log_progress:
            _log_counter = [0]
            _log_start = [time.time()]

            def _log_callback(
                backend_name: str = "",
                operation: str = "",
                width: int = 0,
                depth: int = 0,
                n_samples: int = 0,
            ) -> None:
                _log_counter[0] += 1
                elapsed = time.time() - _log_start[0]
                print(
                    f"[timing] {_log_counter[0]}/{n_combos} "
                    f"{backend_name} {operation} "
                    f"w={width} d={depth} n={n_samples:,} "
                    f"({elapsed:.0f}s elapsed)",
                    flush=True,
                )

            callback = _log_callback

        if progress_ctx is not None:
            timing_task = progress_ctx.add_task("Timing sweep", total=n_combos)

            def _rich_callback(
                backend_name: str = "",
                operation: str = "",
                width: int = 0,
                depth: int = 0,
                n_samples: int = 0,
            ) -> None:
                desc = (
                    f"[cyan]{backend_name:<8}[/] {operation:<18} "
                    f"w={width:<4} d={depth:<4} n={n_samples:>11,}"
                )
                progress_ctx.update(timing_task, advance=1, description=desc)

            callback = _rich_callback

        timing_results, _ = run_timing_sweep(
            passed_backends, preset, progress_callback=callback
        )

    if progress_ctx is not None:
        progress_ctx.stop()

    if log_progress:
        n_errors = sum(1 for tr in timing_results if tr.error)
        n_ok = len(timing_results) - n_errors
        err_msg = f" ({n_errors} skipped due to errors)" if n_errors else ""
        print(f"[done] Timing sweep complete. {n_ok} results.{err_msg}", flush=True)

    # Format output
    terminal_output = format_compact_output(
        correctness_results, timing_results, skipped,
        hardware_info=hardware_info,
    )
    if verbose:
        terminal_output += "\n" + format_verbose_output(
            correctness_results, timing_results, skipped,
            hardware_info=hardware_info,
        )

    json_data = None
    if output_path:
        json_data = format_json_output(
            correctness_results, timing_results, skipped,
            backend_names=list(backend_instances.keys()),
            hardware_info=hardware_info,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

    return terminal_output, json_data
