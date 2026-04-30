# src/whestbench/profiler.py
"""Profiling engine for flopscope simulation performance.

Benchmarks the two core operations — ``run_mlp`` and
``sample_layer_statistics`` — across a configurable grid of network sizes
and sample counts.

Workflow
--------
1. **Correctness check** — simulation outputs are validated for correct
   shapes and value ranges before any timing is recorded.
2. **Timing sweep** — for every (operation, width, depth, n_samples)
   combination the operation is timed ``n_iterations`` times (default 3).
   Garbage collection is disabled during measurement.  Results are expressed
   as median wall-clock seconds.
3. **Output** — a Rich-formatted terminal table and, optionally, a
   machine-readable JSON file that includes hardware metadata and library
   versions for reproducibility.

CLI usage
---------
::

    whest profile-simulation                        # standard sweep
    whest profile-simulation --preset quick          # fast smoke test
    whest profile-simulation --preset exhaustive     # full matrix
    whest profile-simulation --output results.json   # save JSON report

Presets
-------
``super-quick``
    One width (256), one depth (4), 10 000 samples — sub-second sanity check.
``quick``
    One width (256), two depths, two sample counts — finishes in seconds.
``standard`` *(default)*
    Two widths, three depths, two sample counts — under a minute.
``exhaustive``
    Two widths, three depths, three sample counts (up to 1 M) — thorough,
    finishes in minutes.

See Also
--------
``whest profile-simulation --help`` for the full list of CLI flags.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TextIO, Tuple

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np  # needed for np.__version__ in version reporting

from .domain import MLP
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint
from .presentation.adapters import build_profile_presentation
from .presentation.presenters import render_command_presentation
from .simulation import (
    run_mlp as ref_run_mlp,
)
from .simulation import (
    sample_layer_statistics as ref_sample_layer_statistics,
)


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
        widths=[256],
        depths=[4],
        n_samples_list=[10_000],
    ),
    "quick": PresetConfig(
        widths=[256],
        depths=[4, 128],
        n_samples_list=[10_000, 100_000],
    ),
    "standard": PresetConfig(
        widths=[64, 256],
        depths=[4, 32, 128],
        n_samples_list=[10_000, 100_000],
    ),
    "exhaustive": PresetConfig(
        widths=[64, 256],
        depths=[4, 32, 128],
        n_samples_list=[10_000, 100_000, 1_000_000],
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


def _collect_versions() -> Dict[str, str]:
    """Collect version strings for flopscope and numpy."""
    return {"flopscope": flops.__version__, "numpy": np.__version__}


def correctness_check() -> CorrectnessResult:
    """Pre-flight correctness check for the simulation module.

    Validates both ``run_mlp`` (shape and non-negative outputs) and
    ``sample_layer_statistics`` (correct shapes, non-negative variance).

    Returns:
        A :class:`CorrectnessResult` indicating pass/fail and any error
        details.
    """
    try:
        mlp = sample_mlp(8, 4, fnp.random.default_rng(42))
        inputs = fnp.random.default_rng(123).standard_normal((64, 8)).astype(fnp.float32)

        with flops.BudgetContext(flop_budget=int(1e15), quiet=True):
            result = ref_run_mlp(mlp, inputs)
            assert result.shape == (64, 8), f"Expected shape (64, 8), got {result.shape}"
            assert fnp.all(fnp.asarray(result) >= 0.0), "ReLU outputs must be non-negative"

            means, final_mean, avg_var = ref_sample_layer_statistics(mlp, 1000)
            assert means.shape == (4, 8), f"Expected means shape (4, 8), got {means.shape}"
            assert final_mean.shape == (8,), (
                f"Expected final_mean shape (8,), got {final_mean.shape}"
            )
            assert avg_var >= 0.0, f"Expected non-negative variance, got {avg_var}"

        return CorrectnessResult(backend_name="flopscope", passed=True)

    except Exception as e:
        return CorrectnessResult(backend_name="flopscope", passed=False, error=str(e))


def _random_float32(shape: tuple) -> fnp.ndarray:
    """Generate standard-normal float32 array without a float64 intermediate.

    ``np.random.randn(...).astype(np.float32)`` temporarily holds both
    the float64 and float32 arrays in memory, doubling peak usage.
    Using ``Generator.standard_normal`` with ``dtype`` avoids this.
    """
    return fnp.random.default_rng().standard_normal(shape, dtype=fnp.float32)


# Maximum rows per chunk for timed forward passes.  Keeps peak memory
# under ~2 GB per chunk (500K × 256 × 4 bytes ≈ 512 MB input + output).
_TIMING_CHUNK = 500_000


def _time_run_mlp(mlp: MLP, n_samples: int) -> float:
    """Time run_mlp over n_samples, chunked to bound memory."""
    total = 0.0
    for start in range(0, n_samples, _TIMING_CHUNK):
        n = min(_TIMING_CHUNK, n_samples - start)
        inputs = fnp.array(_random_float32((n, mlp.width)))
        t0 = time.perf_counter()
        with flops.BudgetContext(flop_budget=int(1e15), quiet=True):
            ref_run_mlp(mlp, inputs)
        total += time.perf_counter() - t0
        del inputs
    return total


def _time_sample_layer_statistics(mlp: MLP, n_samples: int) -> float:
    """Time a single sample_layer_statistics call."""
    t0 = time.perf_counter()
    with flops.BudgetContext(flop_budget=int(1e15), quiet=True):
        ref_sample_layer_statistics(mlp, n_samples)
    return time.perf_counter() - t0


def run_timing_sweep(
    preset: PresetConfig,
    n_iterations: int = 3,
    progress_callback: Optional[Any] = None,
    warning_stream: TextIO | None = sys.stdout,
) -> List[TimingResult]:
    """Run the full timing sweep across all parameter combinations.

    For each (width, depth, n_samples, operation) tuple the sweep:

    1. Performs an untimed warmup call.
    2. Disables garbage collection and records *n_iterations* timed calls.
    3. Computes the median time.

    Args:
        preset: The parameter grid to iterate over.
        n_iterations: Number of timed repetitions per combination (default 3).
        progress_callback: Optional callable invoked after each measurement.
            Receives keyword arguments ``operation``, ``width``, ``depth``,
            and ``n_samples``.

    Returns:
        A list of :class:`TimingResult` objects.
    """
    results: List[TimingResult] = []

    operations = ["run_mlp", "sample_layer_statistics"]
    time_fns = {
        "run_mlp": _time_run_mlp,
        "sample_layer_statistics": _time_sample_layer_statistics,
    }

    for width in preset.widths:
        for depth in preset.depths:
            mlp = sample_mlp(width, depth, fnp.random.default_rng(42))
            for n_samples in preset.n_samples_list:
                for op in operations:
                    time_fn = time_fns[op]

                    try:
                        # Warmup
                        warmup_t0 = time.perf_counter()
                        time_fn(mlp, min(n_samples, 1000))
                        warmup_time = time.perf_counter() - warmup_t0

                        gc_was_enabled = gc.isenabled()
                        gc.disable()
                        times = []
                        try:
                            for _ in range(n_iterations):
                                t = time_fn(mlp, n_samples)
                                times.append(t)
                        finally:
                            if gc_was_enabled:
                                gc.enable()

                        median_t = float(fnp.median(fnp.asarray(times)))

                        results.append(
                            TimingResult(
                                backend_name="flopscope",
                                operation=op,
                                width=width,
                                depth=depth,
                                n_samples=n_samples,
                                times=times,
                                median_time=median_t,
                                speedup_vs_numpy=1.0,
                                warmup_time=warmup_time,
                            )
                        )

                    except (MemoryError, Exception) as exc:
                        err_msg = f"{type(exc).__name__}: {exc}"
                        if warning_stream is not None:
                            print(
                                f"[warning] flopscope {op} "
                                f"w={width} d={depth} n={n_samples:,} "
                                f"skipped: {err_msg}",
                                file=warning_stream,
                                flush=True,
                            )
                        results.append(
                            TimingResult(
                                backend_name="flopscope",
                                operation=op,
                                width=width,
                                depth=depth,
                                n_samples=n_samples,
                                times=[],
                                median_time=-1.0,
                                speedup_vs_numpy=0.0,
                                error=err_msg,
                            )
                        )

                    if progress_callback:
                        progress_callback(
                            operation=op,
                            width=width,
                            depth=depth,
                            n_samples=n_samples,
                        )

    return results


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
    import io

    from rich.console import Console
    from rich.table import Table

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

    # Timing table (run_mlp + sample_layer_statistics)
    non_profiled = [
        tr for tr in timing_results if tr.operation in ("run_mlp", "sample_layer_statistics")
    ]
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
    import io
    from collections import defaultdict

    from rich.console import Console
    from rich.table import Table

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

    # Line 2 — Correctness
    if all_failed:
        console.print("No backends passed correctness checks. Use --verbose for error details.")
        return buf.getvalue()

    corr_parts: List[str] = []
    for cr in correctness_results:
        if cr.passed:
            corr_parts.append(f"{cr.backend_name} \u2713")
        else:
            corr_parts.append(f"\u26a0 FAIL: {cr.backend_name} (use --verbose for details)")
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
            baseline_name = (
                "numpy"
                if any(e.backend_name == "numpy" for e in entries)
                else entries[0].backend_name
            )

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
    table.add_column("", justify="center")  # status checkmark

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

            run_mlp_str = f"{run_mlp_t:.4f}s" if run_mlp_t is not None else "\u2014"
            out_stats_str = f"{out_stats_t:.4f}s" if out_stats_t is not None else "\u2014"

            status = "\u2713" if bname in passed_names else "\u2717"

            table.add_row(
                bname,
                format_dims(dim_key[0], dim_key[1], dim_key[2]),
                run_mlp_str,
                out_stats_str,
                status,
            )

    console.print(table)

    # Footer
    console.print()
    console.print("[dim italic]Use --verbose for full timing tables with raw times[/dim italic]")

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
        "backend_versions": _collect_versions(),
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
                **({"error": tr.error} if tr.error else {}),
            }
            for tr in timing_results
        ],
    }


def _build_profile_payload(
    correctness_results: List[CorrectnessResult],
    timing_results: List[TimingResult],
    hardware_info: Dict[str, Any],
    *,
    verbose: bool,
) -> Dict[str, Any]:
    passed_names = {cr.backend_name for cr in correctness_results if cr.passed}
    grouped_rows: Dict[Tuple[str, int, int, int], Dict[str, str]] = {}

    for tr in timing_results:
        if tr.error or tr.backend_name not in passed_names:
            continue
        if tr.operation not in ("run_mlp", "sample_layer_statistics"):
            continue
        key = (tr.backend_name, tr.width, tr.depth, tr.n_samples)
        row = grouped_rows.setdefault(
            key,
            {
                "backend": tr.backend_name,
                "dims": format_dims(tr.width, tr.depth, tr.n_samples),
                "run_mlp": "—",
                "sample_layer_statistics": "—",
            },
        )
        row[tr.operation] = f"{tr.median_time:.4f}s"

    timing_rows = [
        grouped_rows[key]
        for key in sorted(grouped_rows, key=lambda item: (item[1], item[2], item[3], item[0]))
    ]

    return {
        "hardware": hardware_info,
        "correctness": [
            {"backend": cr.backend_name, "passed": cr.passed, "error": cr.error}
            for cr in correctness_results
        ],
        "timing": timing_rows,
        "verbose": verbose,
    }


def run_profile(
    preset_name: str = "standard",
    output_path: Optional[str] = None,
    show_progress: bool = False,
    max_threads: Optional[int] = None,
    verbose: bool = False,
    log_progress: bool = False,
    output_format: str = "rich",
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Run the complete profiling pipeline and return formatted results.

    This is the main entry point used by ``whest profile-simulation``.
    It runs a correctness check, performs the timing sweep, and formats
    the output.

    Args:
        preset_name: One of ``"quick"``, ``"standard"``, or ``"exhaustive"``.
            Controls the size of the parameter grid.  See module docstring
            for details.
        output_path: Optional file path to write a JSON report.  The file
            includes hardware info and library versions for reproducibility.
        show_progress: When ``True``, display a Rich progress bar in the
            terminal during timing sweeps.
        max_threads: If set, cap BLAS to at most this many CPU threads.
        log_progress: When ``True``, print one line per benchmark step to
            stdout.  Designed for non-TTY environments (e.g. containers)
            where the Rich progress bar is invisible.
        output_format: Output format, one of ``"rich"``, ``"plain"``, or ``"json"``.

    Returns:
        A tuple of ``(terminal_output, json_data)`` where *terminal_output*
        is a human-formatted string and *json_data* is the JSON dict (or
        ``None`` if *output_path* was not set).

    Example::

        terminal_output, json_data = run_profile(
            preset_name="quick",
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

    emit_human_output = output_format != "json"

    # Set up progress display
    use_rich_progress = emit_human_output and show_progress and not log_progress
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
    if progress_ctx is not None:
        correctness_task = progress_ctx.add_task("Correctness checks", total=1)
        progress_ctx.start()
        progress_ctx.update(
            correctness_task,
            description="Correctness check [cyan]flopscope[/]",
        )

    if log_progress and emit_human_output:
        print("[correctness] flopscope ...", end=" ", flush=True)

    cr = correctness_check()
    correctness_results = [cr]

    if log_progress and emit_human_output:
        print("PASS" if cr.passed else f"FAIL: {cr.error}", flush=True)
    if progress_ctx is not None:
        progress_ctx.advance(correctness_task)

    # Timing sweep (only if correctness passed)
    timing_results: List[TimingResult] = []
    if cr.passed:
        n_combos = (
            len(preset.widths)
            * len(preset.depths)
            * len(preset.n_samples_list)
            * 2  # run_mlp + sample_layer_statistics
        )

        callback: Any = None

        if log_progress and emit_human_output:
            _log_counter = [0]
            _log_start = [time.time()]

            def _log_callback(
                operation: str = "",
                width: int = 0,
                depth: int = 0,
                n_samples: int = 0,
            ) -> None:
                _log_counter[0] += 1
                elapsed = time.time() - _log_start[0]
                print(
                    f"[timing] {_log_counter[0]}/{n_combos} "
                    f"flopscope {operation} "
                    f"w={width} d={depth} n={n_samples:,} "
                    f"({elapsed:.0f}s elapsed)",
                    flush=True,
                )

            callback = _log_callback

        if progress_ctx is not None:
            timing_task = progress_ctx.add_task("Timing sweep", total=n_combos)

            def _rich_callback(
                operation: str = "",
                width: int = 0,
                depth: int = 0,
                n_samples: int = 0,
            ) -> None:
                desc = f"[cyan]flopscope[/] {operation:<18} w={width:<4} d={depth:<4} n={n_samples:>11,}"
                progress_ctx.update(timing_task, advance=1, description=desc)

            callback = _rich_callback

        timing_results = run_timing_sweep(
            preset,
            progress_callback=callback,
            warning_stream=sys.stderr if output_format == "json" else sys.stdout,
        )

    if progress_ctx is not None:
        progress_ctx.stop()

    if log_progress and emit_human_output:
        n_errors = sum(1 for tr in timing_results if tr.error)
        n_ok = len(timing_results) - n_errors
        err_msg = f" ({n_errors} skipped due to errors)" if n_errors else ""
        print(f"[done] Timing sweep complete. {n_ok} results.{err_msg}", flush=True)

    # Format output
    terminal_output = ""
    if emit_human_output:
        presentation = build_profile_presentation(
            _build_profile_payload(
                correctness_results,
                timing_results,
                hardware_info,
                verbose=verbose,
            )
        )
        terminal_output = render_command_presentation(
            presentation,
            output_format="plain" if output_format == "plain" else "rich",
            force_terminal=output_format == "rich",
        )
        if verbose and output_format == "rich":
            terminal_output += "\n" + format_verbose_output(
                correctness_results,
                timing_results,
                {},
                hardware_info=hardware_info,
            )

    json_data = format_json_output(
        correctness_results,
        timing_results,
        {},
        backend_names=["flopscope"],
        hardware_info=hardware_info,
    )
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

    return terminal_output, json_data
