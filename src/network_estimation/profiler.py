# src/network_estimation/profiler.py
"""Profiling engine for simulation backends.

Benchmarks every available simulation backend (numpy, pytorch, numba, jax,
scipy, cython) head-to-head on the two core operations — ``run_mlp`` and
``output_stats`` — across a configurable grid of network sizes and sample
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
    """Parameter sweep grid for profiling.

    Each preset defines a cross-product of network shapes and sample counts.
    Every (width, depth, n_samples) triple is evaluated for both ``run_mlp``
    and ``output_stats``.

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
        operation: ``"run_mlp"`` or ``"output_stats"``.
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
    """Pre-flight correctness check against the NumPy reference implementation.

    Validates both ``run_mlp`` (exact match, rtol=1e-5, atol=1e-6) and
    ``output_stats`` (statistical match, atol=0.15 for means, relative
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
                            progress_callback(
                                backend_name=backend_name,
                                operation=op,
                                width=width,
                                depth=depth,
                                n_samples=n_samples,
                            )

    return results, numpy_baselines


def format_terminal_table(
    correctness_results: List[CorrectnessResult],
    timing_results: List[TimingResult],
    skipped_backends: Dict[str, str],
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
        if progress_ctx is not None:
            progress_ctx.update(
                correctness_task,
                description=f"Correctness check [cyan]{name}[/]",
            )
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

            def callback(
                backend_name: str = "",
                operation: str = "",
                width: int = 0,
                depth: int = 0,
                n_samples: int = 0,
            ) -> None:
                desc = f"Timing sweep [cyan]{backend_name}[/] {operation} w={width} d={depth} n={n_samples:,}"
                progress_ctx.update(timing_task, advance=1, description=desc)
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
