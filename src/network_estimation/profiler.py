# src/network_estimation/profiler.py
"""Profiling engine for simulation backends."""

from __future__ import annotations

import gc
import json
import platform
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .domain import MLP
from .generation import sample_mlp
from .simulation import (
    output_stats as ref_output_stats,
    run_mlp as ref_run_mlp,
)
from .simulation_backend import SimulationBackend
from .simulation_backends import ALL_BACKEND_NAMES, INSTALL_HINTS, get_available_backends


@dataclass
class PresetConfig:
    """Parameter sweep grid for profiling."""

    widths: List[int]
    depths: List[int]
    n_samples_list: List[int]


PRESETS: Dict[str, PresetConfig] = {
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


@dataclass
class CorrectnessResult:
    backend_name: str
    passed: bool
    error: str = ""


@dataclass
class TimingResult:
    backend_name: str
    operation: str
    width: int
    depth: int
    n_samples: int
    times: List[float]
    median_time: float
    speedup_vs_numpy: float


def _collect_hardware_info() -> Dict[str, Any]:
    """Collect hardware info for the profiling report."""
    import os
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }


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
    """Pre-flight correctness check against NumPy reference."""
    try:
        mlp = sample_mlp(8, 4, np.random.default_rng(42))
        inputs = np.random.default_rng(123).standard_normal((64, 8)).astype(np.float32)

        # Exact match for run_mlp
        ref = ref_run_mlp(mlp, inputs)
        result = backend.run_mlp(mlp, inputs)
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-6)

        # Statistical match for output_stats
        ref_means, ref_final, ref_var = ref_output_stats(mlp, 1000)
        fast_means, fast_final, fast_var = backend.output_stats(mlp, 1000)
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


def _time_output_stats(
    backend: SimulationBackend, mlp: MLP, n_samples: int
) -> float:
    """Time a single output_stats call."""
    t0 = time.perf_counter()
    backend.output_stats(mlp, n_samples)
    return time.perf_counter() - t0


def run_timing_sweep(
    backends: Dict[str, SimulationBackend],
    preset: PresetConfig,
    n_iterations: int = 3,
    progress_callback: Optional[Any] = None,
) -> Tuple[List[TimingResult], Dict[str, List[float]]]:
    """Run timing sweep across all backends and parameter combos.

    Returns:
        results: List of TimingResult for each (backend, operation, params) combo
        numpy_baselines: Dict mapping "op:w:d:n" -> list of numpy times
    """
    results: List[TimingResult] = []
    numpy_baselines: Dict[str, List[float]] = {}

    # Collect numpy baselines first
    numpy_backend = backends.get("numpy")

    operations = ["run_mlp", "output_stats"]

    for width in preset.widths:
        for depth in preset.depths:
            mlp = sample_mlp(width, depth, np.random.default_rng(42))
            for n_samples in preset.n_samples_list:
                for op in operations:
                    key = f"{op}:{width}:{depth}:{n_samples}"

                    for backend_name, backend in backends.items():
                        time_fn = _time_run_mlp if op == "run_mlp" else _time_output_stats

                        # Warmup (untimed, critical for JIT backends)
                        time_fn(backend, mlp, min(n_samples, 1000))

                        # Timed iterations with GC disabled
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
                        ))

                        if progress_callback:
                            progress_callback()

    return results, numpy_baselines


def format_terminal_table(
    correctness_results: List[CorrectnessResult],
    timing_results: List[TimingResult],
    skipped_backends: Dict[str, str],
) -> str:
    """Format results as a Rich table string."""
    from rich.console import Console
    from rich.table import Table
    import io

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=140)

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

    # Timing table
    if timing_results:
        table = Table(title="\nTiming Results", show_lines=True)
        table.add_column("Backend", style="cyan")
        table.add_column("Operation")
        table.add_column("Width", justify="right")
        table.add_column("Depth", justify="right")
        table.add_column("N_Samples", justify="right")
        table.add_column("Median Time (s)", justify="right")
        table.add_column("Speedup vs NumPy", justify="right")
        table.add_column("Status")

        for tr in timing_results:
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


def format_json_output(
    correctness_results: List[CorrectnessResult],
    timing_results: List[TimingResult],
    skipped_backends: Dict[str, str],
    backend_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Format results as a JSON-serializable dict."""
    return {
        "hardware": _collect_hardware_info(),
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
            }
            for tr in timing_results
        ],
    }


def run_profile(
    preset_name: str = "standard",
    backend_filter: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show_progress: bool = False,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Main profiling entry point.

    Returns:
        terminal_output: Rich-formatted string for terminal display
        json_data: JSON dict if output_path is set, else None
    """
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
    progress_ctx: Any = None
    if show_progress:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
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
        cr = correctness_check(backend)
        correctness_results.append(cr)
        if cr.passed:
            passed_backends[name] = backend
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
            * 2  # run_mlp + output_stats
            * len(passed_backends)
        )
        if progress_ctx is not None:
            timing_task = progress_ctx.add_task("Timing sweep", total=n_combos)
            callback = lambda: progress_ctx.advance(timing_task)
        else:
            callback = None
        timing_results, _ = run_timing_sweep(
            passed_backends, preset, progress_callback=callback
        )

    if progress_ctx is not None:
        progress_ctx.stop()

    # Format output
    terminal_output = format_terminal_table(
        correctness_results, timing_results, skipped
    )

    json_data = None
    if output_path:
        json_data = format_json_output(
            correctness_results, timing_results, skipped,
            backend_names=list(backend_instances.keys()),
        )
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

    return terminal_output, json_data
