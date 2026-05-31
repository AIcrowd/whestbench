"""CLI and convenience entrypoints for local scoring runs."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from importlib.resources import files
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    overload,
)

import flopscope.numpy as fnp
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from tqdm import tqdm as classic_tqdm
from tqdm.std import TqdmExperimentalWarning

try:
    from tqdm.rich import tqdm as rich_tqdm
except Exception:  # pragma: no cover - optional at runtime
    rich_tqdm = None

from .aicrowd_client import AIcrowdClient  # module-level for monkeypatch + reuse in `submit`
from .dataset import metadata as _wb_metadata
from .dataset_io import _validate_config_name, _validate_split_name
from .dataset_io import metadata_file_hash as _metadata_file_hash
from .estimators import CombinedEstimator
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint
from .loader import load_estimator_from_path, resolve_estimator_class_metadata
from .packaging import package_submission
from .presentation.adapters import (
    build_error_presentation,
    build_init_presentation,
    build_package_presentation,
    build_run_presentation,
    build_smoke_test_presentation,
    build_validate_presentation,
)
from .presentation.models import StepsSection
from .presentation.output import add_output_format_arguments, resolve_output_format
from .presentation.presenters import render_command_presentation
from .reporting import (
    render_agent_report,
    render_human_context_panels,
    render_human_header,
    render_human_report,
    render_human_results,
    render_smoke_test_next_steps,
)
from .runner import (
    EstimatorEntrypoint,
    LocalRunner,
    ResourceLimits,
    RunnerError,
    SubprocessRunner,
)
from .scoring import ContestSpec, evaluate_estimator, make_contest, validate_predictions
from .sdk import BaseEstimator, SetupContext
from .simulation import sample_layer_statistics_chunk_count

_DEFAULT_ESTIMATOR = CombinedEstimator()
ProgressCallback = Callable[[Dict[str, Any]], None]
_SAMPLING_PROGRESS_PHASE = "sampling_ground_truth"


class _RemovedFlopBudgetAction(argparse.Action):
    """Reject `whest create-dataset --flop-budget`, which used to stamp
    a now-removed `flop_budget` field into dataset metadata. Points the
    user at the live runtime flag.

    The flag was removed in schema 2.3 (issue #23). Keeping this action
    indefinitely costs nothing and improves the UX for anyone landing
    here from a stale doc or LLM-suggested command.
    """

    def __init__(self, option_strings, dest, **kwargs):
        kwargs.setdefault("nargs", "?")
        kwargs.setdefault("default", argparse.SUPPRESS)
        kwargs.setdefault("help", argparse.SUPPRESS)
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            "--flop-budget is no longer accepted on 'create-dataset'. "
            "The FLOP budget is a run-time parameter; pass it to "
            "'whest run --flop-budget' instead. Ground truth in the "
            "dataset is independent of the FLOP budget."
        )


def _default_contest_spec() -> ContestSpec:
    return ContestSpec(
        width=256,
        depth=8,
        n_mlps=10,
        flop_budget=68_000_000_000,
        ground_truth_samples=100 * 100 * 256,
    )


def _resolve_whestbench_version() -> str:
    """Resolve the installed whestbench package version."""

    try:
        return package_version("whestbench")
    except Exception:
        return "unknown"


def _json_payload_with_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload_with_metadata = dict(payload)
    payload_with_metadata["whestbench_version"] = _resolve_whestbench_version()
    return payload_with_metadata


def _is_first_progress_phase(phase: str) -> bool:
    return phase in {"generating", _SAMPLING_PROGRESS_PHASE}


def _mlp_label_from_event(event: Dict[str, Any]) -> str:
    mlp_index = event.get("mlp_index")
    n_mlps = event.get("n_mlps")
    mlp_name = event.get("mlp_name")
    if isinstance(mlp_index, int) and isinstance(n_mlps, int):
        if isinstance(mlp_name, str) and mlp_name:
            return f"MLP {mlp_name} ({mlp_index}/{n_mlps})"
        return f"MLP {mlp_index}/{n_mlps}"
    return ""


def _run_progress_columns() -> tuple[Any, ...]:
    return (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed:.0f}/{task.total:.0f} {task.fields[unit]}"),
        TextColumn("{task.fields[mlp_label]}"),
        TimeElapsedColumn(),
    )


class _PlainRunProgressLogger:
    def __init__(self, *, n_mlps: int, refresh_interval_s: float = 0.5) -> None:
        self._n_mlps = n_mlps
        self._refresh_interval_s = refresh_interval_s
        self._last_sampling_emit_at: Optional[float] = None
        self._last_completed = {"generating": -1, _SAMPLING_PROGRESS_PHASE: -1, "scoring": -1}

    def __call__(self, event: Dict[str, Any]) -> None:
        phase = str(event.get("phase", "scoring"))
        completed = int(event.get("completed", 0))
        if completed == self._last_completed.get(phase):
            return
        self._last_completed[phase] = completed
        total_value = event.get("total")
        total = int(total_value) if isinstance(total_value, int) else self._n_mlps

        if phase == _SAMPLING_PROGRESS_PHASE:
            self._emit_sampling(event, completed=completed, total=total)
            return

        print(f"[run] {phase}: {completed}/{total}", file=sys.stderr)

    def _emit_sampling(self, event: Dict[str, Any], *, completed: int, total: int) -> None:
        now = time.monotonic()
        mlp_completed = event.get("mlp_completed")
        mlp_total = event.get("mlp_total")
        force = completed >= total or (
            isinstance(mlp_completed, int)
            and isinstance(mlp_total, int)
            and mlp_completed >= mlp_total
        )
        if (
            self._last_sampling_emit_at is not None
            and not force
            and now - self._last_sampling_emit_at < self._refresh_interval_s
        ):
            return
        self._last_sampling_emit_at = now
        unit = str(event.get("unit") or "chunks")
        mlp_label = _mlp_label_from_event(event)
        suffix = f" ({mlp_label})" if mlp_label else ""
        print(
            f"[run] {_SAMPLING_PROGRESS_PHASE}: {completed}/{total} {unit}{suffix}",
            file=sys.stderr,
        )


def _default_resource_limits() -> ResourceLimits:
    return ResourceLimits(
        setup_timeout_s=5.0,
        predict_timeout_s=30.0,
        memory_limit_mb=65_536,
        flop_budget=68_000_000_000,
        cpu_time_limit_s=None,
        wall_time_limit_s=60.0,
    )


def _debugger_active() -> bool:
    """Detect whether the user has opted into a debugger workflow.

    Returns True if any of:
    - ``sys.gettrace()`` returns a non-None trace function (e.g. running
      under ``python -m pdb`` or an attached debugger).
    - ``PYTHONBREAKPOINT`` is set to a non-empty, non-"0" value (CPython's
      standard env var that controls ``breakpoint()``).
    """
    if sys.gettrace() is not None:
        return True
    pb = os.environ.get("PYTHONBREAKPOINT", "").strip()
    if pb and pb != "0":
        return True
    return False


@overload
def run_default_score(profile: Literal[False] = False) -> float: ...


@overload
def run_default_score(profile: Literal[True]) -> "tuple[float, list[dict[str, Any]]]": ...


def run_default_score(profile: bool = False) -> "Any":
    """Run the built-in smoke-test scenario and return score-only output.

    When ``profile`` is true, this returns ``(score, profile_calls)``.
    """
    spec = _default_contest_spec()
    data = make_contest(spec)
    result = evaluate_estimator(_DEFAULT_ESTIMATOR, data)
    score = result["adjusted_final_layer_score"]
    if profile:
        return score, list(result.get("per_mlp", []))
    return score


def _smoke_test_contest_spec() -> ContestSpec:
    """Lightweight spec for the smoke test.

    Matches the competition shape (width=256, depth=8, flop_budget=6.8e10
    per ContestSpec defaults) so participants exercising the smoke path
    hit the same code paths as the real grader. Only n_mlps and
    ground_truth_samples are scaled down so the smoke runs in well under
    a second — accuracy of the resulting score is not meaningful, this is
    a plumbing check.

    Local timing on a typical dev box: ~0.2s total (ground truth ~0.15s,
    evaluation ~0.03s, CombinedEstimator ~3% budget utilization).
    """
    return ContestSpec(
        width=256,
        depth=8,
        n_mlps=3,
        flop_budget=68_000_000_000,
        ground_truth_samples=10_000,
    )


def run_default_report(
    *,
    profile: bool = False,
    detail: str = "raw",
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Run the built-in smoke-test scenario and return report payload."""
    import time as _time

    spec = _smoke_test_contest_spec()
    if progress is not None:
        data = make_contest(
            spec,
            on_mlp_done=lambda i: progress(
                {"phase": "ground_truth", "completed": i, "total": spec.n_mlps}
            ),
        )
    else:
        data = make_contest(spec)
    t0 = _time.time()
    result = evaluate_estimator(_DEFAULT_ESTIMATOR, data)
    elapsed = _time.time() - t0
    if progress is not None:
        progress({"phase": "scoring", "completed": spec.n_mlps, "total": spec.n_mlps})
    return {
        "schema_version": "1.1",
        "mode": "human",
        "results": result,
        "run_meta": {
            "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_duration_s": elapsed,
            "host": _host_metadata(),
        },
        "run_config": {
            "width": spec.width,
            "depth": spec.depth,
            "n_mlps": spec.n_mlps,
            "seed": spec.seed,
            "flop_budget": spec.flop_budget,
            "wall_time_limit_s": spec.wall_time_limit_s,
            "residual_wall_time_limit_s": spec.residual_wall_time_limit_s,
        },
    }


def _render_plain_text_report(
    report: Dict[str, Any],
    *,
    debug: bool = False,
    command: str = "run",
    include_epilogues: bool = True,
    include_diagnostic_plots_tip: bool = True,
    include_context: bool = True,
) -> str:
    """Render a plain-text report when Rich rendering is unavailable."""
    doc = (
        build_smoke_test_presentation(report, debug=debug)
        if command == "smoke-test"
        else build_run_presentation(report, debug=debug)
    )
    if not include_epilogues:
        doc = replace(doc, epilogue_messages=[])
    elif not include_diagnostic_plots_tip:
        doc = replace(
            doc,
            epilogue_messages=[
                message
                for message in doc.epilogue_messages
                if message != "Use --show-diagnostic-plots to include diagnostic plot panes."
            ],
        )
    if command == "smoke-test":
        return render_human_report(
            report,
            debug=debug,
            presentation_doc=doc,
            output_format="plain",
        )
    return render_human_results(
        report,
        debug=debug,
        presentation_doc=doc,
        output_format="plain",
        include_context=include_context,
        include_epilogues=bool(doc.epilogue_messages),
    )


def _host_metadata() -> Dict[str, Any]:
    return collect_hardware_fingerprint()


def _fmt_ram(value: Optional[int]) -> str:
    """Format byte count as human-readable string for warning messages."""
    if value is None:
        return "unknown RAM"
    gb = value / (1024**3)
    return f"{gb:.0f}GB"


def _pre_run_report(
    *,
    n_mlps: int,
    contest_spec: ContestSpec,
    profile: bool,
    detail: str,
    estimator_class: str,
    estimator_path: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.1",
        "mode": "human",
        "detail": detail,
        "run_meta": {
            "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_finished_at_utc": "n/a",
            "run_duration_s": None,
            "host": _host_metadata(),
        },
        "run_config": {
            "n_mlps": int(n_mlps),
            "width": int(contest_spec.width),
            "depth": int(contest_spec.depth),
            "seed": contest_spec.seed,
            "flop_budget": int(contest_spec.flop_budget),
            "wall_time_limit_s": contest_spec.wall_time_limit_s,
            "residual_wall_time_limit_s": contest_spec.residual_wall_time_limit_s,
            "profile_enabled": bool(profile),
            "estimator_class": estimator_class,
            "estimator_path": estimator_path,
        },
        "results": {},
    }


def _print_human_startup(
    pre_report: Dict[str, Any],
    *,
    estimator_class: str,
    estimator_path: str,
) -> None:
    _print_human_header_and_hints()
    run_config = pre_report.get("run_config")
    if isinstance(run_config, dict):
        run_config["estimator_class"] = estimator_class
        run_config["estimator_path"] = estimator_path

    print(render_human_context_panels(pre_report), end="")


def _print_human_header_and_hints() -> None:
    print(render_human_header(), end="")


def _merge_pre_run_context(report: Dict[str, Any], pre_report: Dict[str, Any]) -> Dict[str, Any]:
    run_meta = report.get("run_meta")
    if not isinstance(run_meta, dict):
        run_meta = {}
        report["run_meta"] = run_meta
    pre_run_meta = pre_report.get("run_meta")
    if isinstance(pre_run_meta, dict):
        started = pre_run_meta.get("run_started_at_utc")
        if started not in {None, "", "n/a"}:
            run_meta["run_started_at_utc"] = started
        pre_host = pre_run_meta.get("host")
        host_meta = run_meta.get("host")
        if isinstance(pre_host, dict):
            if not isinstance(host_meta, dict):
                host_meta = {}
            merged_host = dict(pre_host)
            merged_host.update(host_meta)
            run_meta["host"] = merged_host

    run_config = report.get("run_config")
    if not isinstance(run_config, dict):
        run_config = {}
        report["run_config"] = run_config
    pre_run_config = pre_report.get("run_config")
    if isinstance(pre_run_config, dict):
        for key, value in pre_run_config.items():
            run_config.setdefault(key, value)

    return report


class _LiveTopPaneSession:
    def __init__(
        self,
        pre_report: Dict[str, Any],
        total: int,
        n_mlps: int,
        gen_label: str = "Generating MLPs",
    ) -> None:
        self._pre_report = pre_report
        self._progress = Progress(*_run_progress_columns())
        self._gen_task_id = self._progress.add_task(
            gen_label,
            total=total,
            unit="chunks" if gen_label == "Sampling Ground Truth" else "mlp",
            mlp_label="",
        )
        # Scoring task is added hidden and unstarted so its elapsed timer
        # doesn't tick during the generating phase. It is revealed and
        # started on the first "scoring" progress event.
        self._scoring_task_id = self._progress.add_task(
            "Scoring",
            total=n_mlps,
            start=False,
            visible=False,
            unit="eval",
            mlp_label="",
        )
        self._scoring_started = False
        self._last_completed = {"generating": -1, "scoring": -1}
        self._refresh_interval_s = 0.5
        self._last_refresh_at = 0.0
        self._live = Live(
            self._renderable(),
            console=None,
            auto_refresh=False,
            transient=False,
        )
        self._prev_breakpointhook: Optional[Callable[..., Any]] = None

    def _renderable(self) -> Panel:
        return Panel(
            self._progress,
            title="[bold bright_yellow]Progress[/]",
            border_style="bright_yellow",
        )

    def _refresh(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or (now - self._last_refresh_at) >= self._refresh_interval_s:
            self._last_refresh_at = now
            refresh = getattr(self._live, "refresh", None)
            if callable(refresh):
                refresh()

    def on_progress(self, event: Dict[str, Any]) -> None:
        phase = str(event.get("phase", "scoring"))
        state_phase = "generating" if _is_first_progress_phase(phase) else phase
        completed = int(event.get("completed", 0))
        if completed <= self._last_completed.get(state_phase, -1):
            return
        self._last_completed[state_phase] = completed
        total_value = event.get("total")
        force_refresh = False
        if _is_first_progress_phase(phase):
            total_update = int(total_value) if isinstance(total_value, int) else None
            self._progress.update(
                self._gen_task_id,
                completed=completed,
                total=total_update,
                unit=str(event.get("unit") or "mlp"),
                mlp_label=_mlp_label_from_event(event),
                refresh=False,
            )
            force_refresh = total_update is not None and completed >= total_update
        else:
            if not self._scoring_started:
                self._progress.start_task(self._scoring_task_id)
                self._progress.update(self._scoring_task_id, visible=True, refresh=False)
                self._scoring_started = True
                force_refresh = True
            self._progress.update(self._scoring_task_id, completed=completed, refresh=False)
            if isinstance(total_value, int) and completed >= total_value:
                force_refresh = True
        self._refresh(force=force_refresh)

    def update_run_meta(self, run_meta: Dict[str, Any]) -> None:
        # The human context panels are printed once outside the Live region, so
        # final run metadata is not re-rendered in Rich mode anymore. Keep this
        # hook as a compatibility seam for tests and future display changes.
        self._pre_report["run_meta"] = run_meta

    def start(self) -> None:
        # Capture the current breakpointhook at start-time so we compose with
        # any user-installed hook (e.g. IPython) rather than overwriting it.
        # When breakpoint() fires from inside predict(), we stop Rich's Live
        # overlay first so the debugger's prompt and input are visible.
        self._prev_breakpointhook = sys.breakpointhook

        def _bphook(*args: Any, **kwargs: Any) -> Any:
            # Capture the prev hook locally before stop() nils the attribute,
            # so we can still delegate to it after tearing down the overlay.
            prev = self._prev_breakpointhook
            self.stop()
            if prev is None:
                return None
            return prev(*args, **kwargs)

        sys.breakpointhook = _bphook
        self._live.start()
        refresh = getattr(self._live, "refresh", None)
        if callable(refresh):
            refresh()
        self._last_refresh_at = time.monotonic()

    def stop(self) -> None:
        self._live.stop()
        # Restore the previous breakpointhook. The hook wrapper also restores
        # on first fire; this branch covers the normal "Live ended without a
        # breakpoint" path. Guard against double-stop with the None sentinel.
        if self._prev_breakpointhook is not None:
            sys.breakpointhook = self._prev_breakpointhook
            self._prev_breakpointhook = None
        print()


@contextmanager
def _live_top_pane_session(
    pre_report: Dict[str, Any],
    total: int,
    n_mlps: int,
    gen_label: str = "Generating MLPs",
) -> Iterator[_LiveTopPaneSession]:
    session = _LiveTopPaneSession(pre_report, total, n_mlps, gen_label=gen_label)
    session.start()
    try:
        yield session
    finally:
        session.stop()


@contextmanager
def _progress_callback(
    total: int, n_mlps: int, gen_label: str = "Generating MLPs"
) -> Iterator[ProgressCallback]:
    if rich_tqdm is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", TqdmExperimentalWarning)
            gen_bar = classic_tqdm(
                total=total,
                desc=gen_label,
                unit="chunk" if gen_label == "Sampling Ground Truth" else "mlp",
                file=sys.stdout,
                position=0,
            )
        # Scoring bar is constructed on first scoring event so its elapsed
        # timer doesn't tick during the generating phase.
        bars: Dict[str, Any] = {"scoring": None}

        state = {"gen": 0, "scoring": 0}

        def _on_progress(event: Dict[str, Any]) -> None:
            phase = str(event.get("phase", "scoring"))
            completed = int(event.get("completed", 0))
            if _is_first_progress_phase(phase):
                if isinstance(event.get("total"), int):
                    gen_bar.total = int(event["total"])
                if event.get("unit") == "chunks":
                    gen_bar.unit = "chunk"
                delta = completed - state["gen"]
                if delta > 0:
                    gen_bar.update(delta)
                    state["gen"] = completed
            else:
                if bars["scoring"] is None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", TqdmExperimentalWarning)
                        bars["scoring"] = classic_tqdm(
                            total=total,
                            desc="Scoring",
                            unit="eval",
                            file=sys.stdout,
                            position=1,
                        )
                delta = completed - state["scoring"]
                if delta > 0:
                    bars["scoring"].update(delta)
                    state["scoring"] = completed

        try:
            yield _on_progress
        finally:
            gen_bar.close()
            if bars["scoring"] is not None:
                bars["scoring"].close()
            print()
        return

    progress = Progress(*_run_progress_columns())
    gen_task_id = progress.add_task(
        gen_label,
        total=total,
        unit="chunks" if gen_label == "Sampling Ground Truth" else "mlp",
        mlp_label="",
    )
    # Scoring task starts hidden and unstarted — revealed on first scoring event.
    scoring_task_id = progress.add_task(
        "Scoring", total=n_mlps, start=False, visible=False, unit="eval", mlp_label=""
    )
    scoring_started = {"flag": False}
    state = {"generating": -1, "scoring": -1}
    refresh_state = {"last_at": 0.0}
    live = Live(
        Panel(progress, title="[bold bright_yellow]Progress[/]", border_style="bright_yellow"),
        console=None,
        auto_refresh=False,
        transient=False,
    )

    def _refresh(*, force: bool = False) -> None:
        now = time.monotonic()
        if force or (now - refresh_state["last_at"]) >= 0.5:
            refresh_state["last_at"] = now
            refresh = getattr(live, "refresh", None)
            if callable(refresh):
                refresh()

    def _on_progress(event: Dict[str, Any]) -> None:
        phase = str(event.get("phase", "scoring"))
        state_phase = "generating" if _is_first_progress_phase(phase) else phase
        completed = int(event.get("completed", 0))
        if completed <= state.get(state_phase, -1):
            return
        state[state_phase] = completed
        total_value = event.get("total")
        force_refresh = False
        if _is_first_progress_phase(phase):
            total_update = int(total_value) if isinstance(total_value, int) else None
            progress.update(
                gen_task_id,
                completed=completed,
                total=total_update,
                unit=str(event.get("unit") or "mlp"),
                mlp_label=_mlp_label_from_event(event),
                refresh=False,
            )
            force_refresh = total_update is not None and completed >= total_update
        else:
            if not scoring_started["flag"]:
                progress.start_task(scoring_task_id)
                progress.update(scoring_task_id, visible=True, refresh=False)
                scoring_started["flag"] = True
                force_refresh = True
            progress.update(scoring_task_id, completed=completed, refresh=False)
            if isinstance(total_value, int) and completed >= total_value:
                force_refresh = True
        _refresh(force=force_refresh)

    try:
        live.start()
        refresh = getattr(live, "refresh", None)
        if callable(refresh):
            refresh()
        refresh_state["last_at"] = time.monotonic()
        yield _on_progress
    finally:
        live.stop()
        print()


def _write_init_template(target_dir: Path) -> "list[str]":
    created: "list[str]" = []
    target_dir.mkdir(parents=True, exist_ok=True)

    estimator_file = target_dir / "estimator.py"
    if not estimator_file.exists():
        template = files("whestbench") / "templates" / "estimator.py.tmpl"
        estimator_file.write_text(template.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        created.append(str(estimator_file))

    requirements_file = target_dir / "requirements.txt"
    if not requirements_file.exists():
        requirements_file.write_text("# Optional custom dependencies\n", encoding="utf-8")
        created.append(str(requirements_file))

    return created


def _run_validate_checks(
    estimator_path: "Any",
    *,
    class_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    estimator, metadata = load_estimator_from_path(estimator_path, class_name=class_name)
    context = SetupContext(
        width=4,
        depth=2,
        flop_budget=100,
        api_version="1.0",
        seed=seed if seed is not None else 0,
    )
    mlp = sample_mlp(width=4, depth=2)
    checks: list[dict[str, str]] = []
    try:
        checks.append({"name": "class resolved", "status": "ok", "detail": metadata.class_name})
        estimator.setup(context)
        checks.append({"name": "setup(context) completed", "status": "ok", "detail": "ok"})
        predictions = estimator.predict(mlp, 100)
        arr = validate_predictions(predictions, depth=mlp.depth, width=mlp.width)
        checks.append(
            {
                "name": "predict() returned shape",
                "status": "ok",
                "detail": str(tuple(arr.shape)),
            }
        )
        checks.append({"name": "values finite", "status": "ok", "detail": "all finite"})
    finally:
        estimator.teardown()
    return {
        "ok": True,
        "class_name": metadata.class_name,
        "module_name": metadata.module_name,
        "output_shape": list(arr.shape),
        "checks": checks,
    }


def validate_submission_entrypoint(
    estimator_path: "Any",
    *,
    class_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    result = _run_validate_checks(estimator_path, class_name=class_name, seed=seed)
    return {
        "ok": result["ok"],
        "class_name": result["class_name"],
        "module_name": result["module_name"],
        "output_shape": result["output_shape"],
    }


def _aicrowd_verify_identity(api_key: str) -> Dict[str, Any]:
    """Validate an AIcrowd API key and return identity info ({"id": ...}).

    Raises AIcrowdAPIError (or a transport error) on failure. Kept module-level
    so tests can monkeypatch it without hitting the network.
    """
    pid = AIcrowdClient(api_key=api_key).verify_identity()
    return {"id": pid}


def _build_participant_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whest",
        description=(
            "Participant-first WhestBench CLI. Starter examples live in https://github.com/AIcrowd/whest-starterkit."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_test_parser = subparsers.add_parser(
        "smoke-test",
        help=(
            "Run a built-in CombinedEstimator dashboard check and print next steps "
            "for participant workflows."
        ),
    )
    smoke_test_parser.add_argument("--detail", choices=("raw", "full"), default="raw")
    smoke_test_parser.add_argument("--profile", action="store_true")
    smoke_test_parser.add_argument("--show-diagnostic-plots", action="store_true")
    smoke_test_parser.add_argument("--debug", action="store_true")
    add_output_format_arguments(smoke_test_parser)
    smoke_test_parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        metavar="N",
        help="Limit BLAS to at most N CPU threads.",
    )

    version_parser = subparsers.add_parser("version", help="Print whestbench version.")
    add_output_format_arguments(version_parser)

    init_parser = subparsers.add_parser("init", help="Create starter estimator files.")
    init_parser.add_argument("path", nargs="?", default=".")
    init_parser.add_argument("--debug", action="store_true")
    add_output_format_arguments(init_parser)

    validate_parser = subparsers.add_parser("validate", help="Validate estimator contract.")
    validate_parser.add_argument(
        "--estimator",
        required=True,
        help="Path to estimator.py (see https://github.com/AIcrowd/whest-starterkit for starter files).",
    )
    validate_parser.add_argument("--class", dest="class_name")
    validate_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the validation run. Seeds estimator setup via ctx.seed. Default: omitted (ctx.seed = 0).",
    )
    validate_parser.add_argument("--debug", action="store_true")
    add_output_format_arguments(validate_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Run local evaluation for an estimator.",
        epilog=(
            "Dataset usage: pass --dataset <local-dir> or hf://owner/repo[@rev]. "
            "See docs/guides/datasets.md for the full lifecycle."
        ),
    )
    run_parser.add_argument(
        "--estimator",
        required=True,
        help="Path to estimator.py (see https://github.com/AIcrowd/whest-starterkit for starter files).",
    )
    run_parser.add_argument("--class", dest="class_name")
    run_parser.add_argument(
        "--runner", choices=("local", "subprocess", "server", "inprocess"), default="local"
    )
    run_parser.add_argument(
        "--n-mlps",
        type=int,
        default=None,
        help=(
            "Number of MLPs to evaluate. Default: 10 when --dataset is not "
            "provided; otherwise the full dataset size. Clamped to the dataset "
            "size when --dataset is set and --n-mlps exceeds it."
        ),
    )
    run_parser.add_argument("--detail", choices=("raw", "full"), default="raw")
    run_parser.add_argument("--profile", action="store_true")
    run_parser.add_argument("--show-diagnostic-plots", action="store_true")
    add_output_format_arguments(run_parser)
    run_parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a baked dataset directory, or hf://owner/repo[@revision] for HF Hub.",
    )
    run_parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help=(
            "Stream the dataset from HF instead of downloading it. Iteration-only "
            "(no random access). Data is NOT cached — subsequent runs will re-fetch. "
            "Useful for small --n-mlps debugging runs. See "
            "docs/guides/datasets.md#streaming-mode."
        ),
    )
    run_parser.add_argument(
        "--revision",
        default=None,
        help="HF Hub revision (tag or commit SHA) for --dataset.",
    )
    run_parser.add_argument(
        "--split",
        default=None,
        type=_validate_split_name,
        help=(
            "For multi-split datasets, the split to evaluate. Required when the dataset "
            "is multi-split; optional when single-split (defaults to the only split)."
        ),
    )
    run_parser.add_argument(
        "--flop-budget",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Effective compute budget per MLP in FLOPs. Caps "
            "C_m = F_m + lambda*R_m (analytical FLOPs plus charged residual "
            "wall time). Always honored; any flop_budget stored in "
            "--dataset's metadata is ignored. "
            "Default: 68_000_000_000 (6.8e10)."
        ),
    )
    run_parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        metavar="N",
        help="Ground truth samples per MLP (default: width*width*256). Lower values speed up generation at the cost of noisier scores.",
    )
    run_parser.add_argument("--debug", action="store_true")
    run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop on the first estimator error and let the raw Python traceback "
            "propagate (combine with --debug to show it)."
        ),
    )
    run_parser.add_argument(
        "--wall-time-limit",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Wall-clock time limit per predict call (default: 60.0 seconds).",
    )
    run_parser.add_argument(
        "--residual-wall-time-limit",
        dest="residual_wall_time_limit",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Time limit for non-flopscope operations per predict call (default: unlimited).",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for the run. Without --dataset, seeds both MLP generation "
            "and estimator setup. With --dataset, MLP seeds come from the dataset; "
            "this flag seeds estimator setup only. Default: omitted "
            "(ctx.seed defaults to 0; run_config.seed is null in the JSON output)."
        ),
    )
    run_parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        metavar="N",
        help="Limit BLAS to at most N CPU threads.",
    )

    # Deprecated: redirect to `whest dataset bake`
    create_ds_parser = subparsers.add_parser("create-dataset", help=argparse.SUPPRESS)
    create_ds_parser.add_argument("--n-mlps", type=int, default=None)
    create_ds_parser.add_argument("--n-samples", type=int, default=None)
    create_ds_parser.add_argument("--width", type=int, default=None)
    create_ds_parser.add_argument("--depth", type=int, default=None)
    create_ds_parser.add_argument("--seed", type=int, default=None)
    create_ds_parser.add_argument("-o", "--output", "--output-path", default=None)
    create_ds_parser.add_argument("--debug", action="store_true")
    create_ds_parser.add_argument("--max-threads", type=int, default=None)
    create_ds_parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    create_ds_parser.add_argument("--flop-budget", action=_RemovedFlopBudgetAction)
    add_output_format_arguments(create_ds_parser)

    # ----- whest dataset {bake,push,pull,merge,inspect} -----
    dataset_parser = subparsers.add_parser(
        "dataset", help="Dataset bake/publish/load/merge/inspect commands."
    )
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_cmd", required=True)

    bake_p = dataset_sub.add_parser(
        "bake",
        help="Bake a new dataset to a directory.",
        epilog="See docs/guides/datasets.md for a complete walk-through.",
    )
    bake_p.add_argument(
        "--n-mlps", type=int, required=True, help="Total number of MLPs in the logical dataset."
    )
    bake_p.add_argument("--n-samples", type=int, required=True)
    bake_p.add_argument("--width", type=int, required=True)
    bake_p.add_argument("--depth", type=int, required=True)
    bake_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # legacy 2.0 flag; hidden, rejected at dispatch with migration hint
    )
    bake_p.add_argument(
        "--mlp-seeds",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing an array of N explicit per-MLP seeds "
            "(each a non-negative int < 2**63). If omitted, auto-generate via "
            "secrets.randbits(63). See docs/reference/dataset-format.md."
        ),
    )

    def _split_name_arg(value: str) -> str:
        try:
            return _validate_split_name(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    def _config_name_arg(value: str) -> str:
        try:
            return _validate_config_name(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    bake_p.add_argument(
        "--split",
        default="public",
        type=_split_name_arg,
        help="Split name. Must match [a-z][a-z0-9-]* (HF Hub split-name convention).",
    )
    bake_p.add_argument(
        "--config",
        default="default",
        type=_config_name_arg,
        help=(
            "HF dataset config name for this split. Defaults to 'default'. "
            "Use this when authoring config-per-split datasets."
        ),
    )
    bake_p.add_argument("--output", required=True, help="Output directory (must not exist).")
    bake_p.add_argument("--torch", action="store_true", help="Use GPU/torch backend.")
    bake_p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    bake_p.add_argument("--mlps-per-batch", type=int, default=None)
    bake_p.add_argument("--chunk-size", type=int, default=None)
    bake_p.add_argument("--slice", dest="slice_spec", help="K/N — this slice K of N (0-indexed).")
    bake_p.add_argument(
        "--mlp-range", dest="mlp_range_str", help="START-END (inclusive on both ends), e.g. 0-249."
    )

    upload_p = dataset_sub.add_parser(
        "upload",
        aliases=["push"],
        help="Upload a baked dataset to HF Hub.",
        epilog="See docs/guides/datasets.md for a complete walk-through.",
    )
    upload_p.add_argument("local_dir")
    upload_p.add_argument("--repo", required=True, help="HF repo id (org/name).")
    upload_p.add_argument("--tag", default=None, help="Optional git tag (e.g. v1).")
    upload_p.add_argument("--private", action="store_true")
    upload_p.add_argument("--token", default=None)
    upload_p.add_argument("--message", default=None)

    download_p = dataset_sub.add_parser(
        "download",
        aliases=["pull"],
        help="Download a dataset from HF Hub.",
        epilog="See docs/guides/datasets.md for a complete walk-through.",
    )
    download_p.add_argument("repo_id")
    download_p.add_argument("--revision", default=None)
    download_p.add_argument("--output", required=True)
    download_p.add_argument("--token", default=None)
    download_p.add_argument(
        "--split",
        default=None,
        type=_validate_split_name,
        help="Optional: download only the specified split's parquet (and metadata/README).",
    )

    merge_p = dataset_sub.add_parser(
        "merge",
        help="Merge partial bakes into one dataset.",
        epilog="See docs/guides/datasets.md for a complete walk-through.",
    )
    merge_p.add_argument("inputs", nargs="+", help="Partial dataset directories.")
    merge_p.add_argument("--output", required=True)

    info_p = dataset_sub.add_parser(
        "info",
        aliases=["inspect"],
        help="Print dataset metadata.",
        epilog="See docs/guides/datasets.md for a complete walk-through.",
    )
    info_p.add_argument("source", help="Local dir or HF repo id.")
    info_p.add_argument("--revision", default=None)

    combine_p = dataset_sub.add_parser(
        "combine-splits",
        help="Combine N single-split datasets into a multi-split dataset directory.",
        epilog="See docs/guides/datasets.md for a complete walk-through.",
    )
    combine_p.add_argument(
        "input_dirs",
        nargs="+",
        help="One or more complete single-split dataset directories.",
    )
    combine_p.add_argument(
        "--output",
        required=True,
        help="Output directory (must not exist).",
    )
    combine_p.add_argument(
        "--default-split",
        default=None,
        help=(
            "Optional name of the split that downstream consumers should fall "
            "back to when --split is omitted on a multi-split dataset. Must "
            "match one of the input splits. Recorded as 'default_split' in "
            "the combined metadata.json and used by `whest run`."
        ),
    )
    combine_p.add_argument(
        "--skip-prepared-arrow",
        action="store_true",
        help=(
            "Skip generation of prepared/<split>/ Arrow artifacts. By default "
            "combine-splits emits `Dataset.save_to_disk()` directories for "
            "each split so `whestbench.load_dataset` can memory-map them "
            "directly on the consumer side (no parquet→arrow conversion). "
            "Skip if the prepare cost outweighs the runtime win for your use."
        ),
    )

    prepare_p = dataset_sub.add_parser(
        "prepare-arrow",
        help=(
            "Patch an existing multi-split dataset directory with "
            "prepared/<split>/ Arrow artifacts so consumers can skip the "
            "parquet→arrow conversion on cold cache."
        ),
        epilog=(
            "Use this to upgrade a dataset baked before this feature existed "
            "without re-running the whole bake. Idempotent — re-runs replace "
            "any existing prepared/ subtree."
        ),
    )
    prepare_p.add_argument(
        "dataset_dir",
        help="Path to an existing multi-split dataset directory (with data/, metadata.json).",
    )

    package_parser = subparsers.add_parser("package", help="Package submission artifact.")
    package_parser.add_argument("--estimator", required=True)
    package_parser.add_argument("--class", dest="class_name")
    package_parser.add_argument("--requirements")
    package_parser.add_argument("--submission-metadata")
    package_parser.add_argument("--approach")
    package_parser.add_argument("--output")
    package_parser.add_argument("--debug", action="store_true")
    add_output_format_arguments(package_parser)

    profile_parser = subparsers.add_parser(
        "profile-simulation",
        help="Benchmark flopscope simulation performance.",
    )
    profile_parser.add_argument(
        "--preset",
        choices=("super-quick", "quick", "standard", "exhaustive"),
        default="standard",
        help="Parameter sweep preset (default: standard).",
    )
    profile_parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results.",
    )
    profile_parser.add_argument("--debug", action="store_true")
    add_output_format_arguments(profile_parser)
    profile_parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        metavar="N",
        help="Limit BLAS to at most N CPU threads.",
    )
    profile_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show full timing tables with all columns and raw data.",
    )
    profile_parser.add_argument(
        "--log-progress",
        action="store_true",
        default=False,
        help="Print one line per benchmark step (for non-TTY environments like containers).",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run install/environment health checks.",
    )
    add_output_format_arguments(doctor_parser)
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures for exit-code purposes.",
    )
    doctor_parser.add_argument(
        "--debug",
        action="store_true",
        help="Re-raise exceptions from crashing checks instead of capturing them.",
    )

    login_parser = subparsers.add_parser(
        "login",
        help="Store your AIcrowd API key (interoperable with aicrowd-cli).",
    )
    login_parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="AIcrowd API key (from your AIcrowd profile page). Prompted if omitted.",
    )
    add_output_format_arguments(login_parser)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit to AIcrowd (packages an estimator if needed, then uploads).",
        epilog=(
            "Auth comes from `whest login` (or AICROWD_API_KEY / --api-key). "
            "The default challenge is arc-white-box-estimation-challenge-2026."
        ),
    )
    submit_grp = submit_parser.add_mutually_exclusive_group(required=True)
    submit_grp.add_argument("artifact", nargs="?", help="Path to a submission .tar.gz.")
    submit_grp.add_argument("--estimator", help="Path to estimator.py (packaged before submit).")
    submit_parser.add_argument("--class", dest="class_name", help="Estimator class name.")
    submit_parser.add_argument("--requirements", help="requirements.txt for packaging.")
    submit_parser.add_argument("--submission-metadata", help="submission.yaml for packaging.")
    submit_parser.add_argument("--approach", help="approach.md for packaging.")
    submit_parser.add_argument(
        "--challenge",
        default="arc-white-box-estimation-challenge-2026",
        help="Challenge slug (default: arc-white-box-estimation-challenge-2026).",
    )
    submit_parser.add_argument("--description", default="Submitted via whest submit")
    submit_parser.add_argument("--api-key", dest="api_key", default=None, help=argparse.SUPPRESS)
    submit_parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll AIcrowd submission status until graded/failed.",
    )
    add_output_format_arguments(submit_parser)

    return parser


def _normalize_runner_name(raw_runner: str) -> str:
    """Normalize legacy/alias runner names to canonical internal names."""
    if raw_runner == "server":
        return "subprocess"
    if raw_runner == "inprocess":
        return "local"
    return raw_runner


def _parse_mlp_range_cli(arg, *, slice_spec, n_mlps):
    """Convert CLI --mlp-range or --slice flag to (start, end) exclusive-end."""
    if arg is not None and slice_spec is not None:
        raise SystemExit("Use either --mlp-range or --slice, not both.")
    if arg is not None:
        try:
            start_s, end_s = arg.split("-")
            start, end_incl = int(start_s), int(end_s)
        except (ValueError, TypeError):
            raise SystemExit(f"--mlp-range must be START-END (inclusive), got {arg!r}")
        return (start, end_incl + 1)
    if slice_spec is not None:
        try:
            k_s, n_s = slice_spec.split("/")
            k, n = int(k_s), int(n_s)
        except (ValueError, TypeError):
            raise SystemExit(f"--slice must be K/N, got {slice_spec!r}")
        if not (0 <= k < n):
            raise SystemExit(f"--slice K/N requires 0 <= K < N, got {slice_spec!r}")
        start = k * n_mlps // n
        end = (k + 1) * n_mlps // n
        return (start, end)
    return None


def _resolve_dataset_arg(arg, *, revision):
    """Resolve ``--dataset`` argument to (repo_or_path, revision, is_local).

    Accepts:
        - Local path: starts with ./, /, ~/, Windows drive letter, or exists on disk
        - hf:// URL: e.g. "hf://owner/repo@v1" or "hf://owner/repo"
        - "owner/repo" with --revision flag explicitly set

    Bare "owner/repo" without --revision is rejected to force explicit pinning.
    """
    from pathlib import Path

    if arg.startswith("hf://"):
        body = arg[len("hf://") :]
        if "@" in body:
            repo, rev = body.split("@", 1)
            return (repo, rev, False)
        return (body, None, False)

    if arg.startswith(("./", "/", "~/")) or Path(arg).exists() or (len(arg) >= 2 and arg[1] == ":"):
        return (arg, revision, True)

    if "/" in arg:
        if revision is None:
            raise SystemExit(
                f"--dataset {arg!r} looks like an HF Hub repo but no revision is "
                f"pinned. Either pass --revision <tag> or use hf://{arg}@<tag>."
            )
        return (arg, revision, False)

    raise SystemExit(f"--dataset {arg!r} not recognized as local path or HF repo.")


_DEPRECATED_DATASET_ALIASES = {"push": "upload", "pull": "download", "inspect": "info"}


def _dispatch_dataset_command(args) -> int:
    import json as _json
    from pathlib import Path as _Path

    sub = args.dataset_cmd
    if sub in _DEPRECATED_DATASET_ALIASES:
        from .ui import say

        canonical = _DEPRECATED_DATASET_ALIASES[sub]
        say.warn(
            f"`whest dataset {sub}` is deprecated; use `whest dataset {canonical}`. "
            f"Aliases will be removed in v0.7."
        )
        sub = canonical

    if sub == "bake":
        import time as _time

        from .dataset import create_dataset as _create_dataset
        from .ui import format_bytes, format_duration, progress_count, say

        # Legacy --seed rejection
        if args.seed is not None:
            print(
                "error: --seed is no longer supported as of seed_protocol 3.0. "
                "Pass --mlp-seeds <file.json> (JSON array of N ints) or omit "
                "the flag for auto-generation. See docs/reference/dataset-format.md.",
                file=sys.stderr,
            )
            return 1

        # Materialise mlp_seeds from --mlp-seeds file (or leave None for auto-gen).
        mlp_seeds = None
        if args.mlp_seeds is not None:
            try:
                with open(args.mlp_seeds, "r") as _f:
                    mlp_seeds = json.load(_f)
            except (OSError, json.JSONDecodeError) as exc:
                print(
                    f"error: cannot read --mlp-seeds file {args.mlp_seeds!r}: invalid JSON: {exc}",
                    file=sys.stderr,
                )
                return 1
            if not isinstance(mlp_seeds, list):
                print(
                    f"error: --mlp-seeds file {args.mlp_seeds!r} must contain a "
                    f"JSON array; got {type(mlp_seeds).__name__}.",
                    file=sys.stderr,
                )
                return 1

        mlp_range = _parse_mlp_range_cli(
            args.mlp_range_str,
            slice_spec=args.slice_spec,
            n_mlps=args.n_mlps,
        )

        say.intent(
            f"Baking {args.n_mlps} MLPs (width={args.width}, depth={args.depth}) → {args.output}"
        )
        _t0 = _time.perf_counter()

        # Phase callback drives one progress_count bar per phase. Each phase's
        # context manager is opened on its first event and closed in `finally`
        # below so a mid-bake exception still tears the bars down cleanly.
        _bake_phase_labels = {
            "generating": "Generating MLPs",
            "sampling": "Sampling ground truth",
        }
        _bars: Dict[str, tuple[Any, Any]] = {}

        def _on_bake_progress(event: Dict[str, Any]) -> None:
            phase = event.get("phase")
            if not isinstance(phase, str):
                return
            total = int(event.get("total", 0) or 0)
            completed = int(event.get("completed", 0) or 0)
            if phase not in _bars:
                label = _bake_phase_labels.get(phase, phase)
                ctx = progress_count(total=total, label=label)
                handle = ctx.__enter__()
                _bars[phase] = (ctx, handle)
            _, handle = _bars[phase]
            handle.update(completed=completed)

        try:
            try:
                if args.torch:
                    from .dataset_torch import create_dataset_torch

                    create_dataset_torch(
                        n_mlps=args.n_mlps,
                        n_samples=args.n_samples,
                        width=args.width,
                        depth=args.depth,
                        mlp_seeds=mlp_seeds,
                        output_path=args.output,
                        split=args.split,
                        config=args.config,
                        mlp_range=mlp_range,
                        device=args.device,
                        mlps_per_batch=args.mlps_per_batch,
                        chunk_size=args.chunk_size,
                        progress=_on_bake_progress,
                    )
                else:
                    _create_dataset(
                        n_mlps=args.n_mlps,
                        n_samples=args.n_samples,
                        width=args.width,
                        depth=args.depth,
                        mlp_seeds=mlp_seeds,
                        output_path=args.output,
                        split=args.split,
                        config=args.config,
                        mlp_range=mlp_range,
                        progress=_on_bake_progress,
                    )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
        finally:
            for _phase, (_ctx, _handle) in _bars.items():
                _ctx.__exit__(None, None, None)

        _elapsed = _time.perf_counter() - _t0
        _out = Path(args.output)
        _size = sum(p.stat().st_size for p in _out.rglob("*") if p.is_file() and not p.is_symlink())
        say.ok(f"Wrote {_out} in {format_duration(_elapsed)} ({format_bytes(_size)})")
        say.hint(f"To publish: `whest dataset upload {_out} --repo aicrowd/<name> --tag v1`")
        return 0

    if sub == "upload":
        import time as _time

        from .hf_progress import hf_upload
        from .hub import publish_dataset
        from .ui import format_bytes, format_duration, say

        _revision_label = args.tag or "main"
        _title = f"hf://{args.repo}@{_revision_label}"
        _console = Console()
        say.intent(f"Uploading {args.local_dir} → {_title}", console=_console)

        # Preflight summary: count files + bytes in the local dir.
        _local = Path(args.local_dir)
        _files = [p for p in _local.rglob("*") if p.is_file() and not p.is_symlink()]
        _total = sum(p.stat().st_size for p in _files)
        _plural = "s" if len(_files) != 1 else ""
        say.step(
            f"Total: {format_bytes(_total)} ({len(_files)} file{_plural}) — "
            f"will create repo if needed.",
            console=_console,
        )

        _t0 = _time.perf_counter()
        with hf_upload(_console, title=_title, local_dir=args.local_dir):
            sha = publish_dataset(
                args.local_dir,
                repo_id=args.repo,
                tag=args.tag,
                token=args.token,
                commit_message=args.message,
                private=args.private,
            )
        _elapsed = _time.perf_counter() - _t0

        say.ok(f"Uploaded {_title} in {format_duration(_elapsed)}", console=_console)
        say.step(f"Commit: {sha}", console=_console)
        say.step(
            f"Visible at: https://huggingface.co/datasets/{args.repo}/tree/{_revision_label}",
            console=_console,
        )
        return 0

    if sub == "download":
        import time as _time

        from huggingface_hub import snapshot_download

        from .hf_progress import hf_download, hf_preflight
        from .ui import format_bytes, format_duration, say

        _split = getattr(args, "split", None)
        _revision_label = args.revision or "main"
        _title = f"hf://{args.repo_id}@{_revision_label}"
        _console = Console()
        say.intent(f"Downloading {_title} → {args.output}", console=_console)

        # Preflight: surface file count, byte total, cache state before any work.
        preflight = hf_preflight(args.repo_id, revision=args.revision, split=_split)
        if preflight is not None:
            _files = preflight.file_count
            _plural = "s" if _files != 1 else ""
            _cache_label = "cached" if preflight.is_cached else "not cached"
            say.step(
                f"Preflight: {_files} file{_plural}, "
                f"{format_bytes(preflight.total_bytes)} — {_cache_label}.",
                console=_console,
            )
            mode: Literal["cache_hit", "materialize"] = (
                "cache_hit" if preflight.is_cached else "materialize"
            )
        else:
            say.step("Preflight: unavailable.", console=_console)
            mode = "materialize"

        allow_patterns = None
        if _split is not None:
            allow_patterns = [
                f"data/{_split}-*.parquet",
                "metadata.json",
                "README.md",
                ".gitattributes",
            ]

        _t0 = _time.perf_counter()
        with hf_download(_console, title=_title, preflight=preflight, mode=mode):
            local = snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                local_dir=args.output,
                token=args.token,
                allow_patterns=allow_patterns,
            )
        _elapsed = _time.perf_counter() - _t0

        # If --split was given, verify the download actually got parquet(s) for it.
        # snapshot_download silently succeeds with zero matches if the glob doesn't
        # match any file in the repo (e.g. typo'd split name or single-split repo).
        if _split is not None:
            matches = list(Path(local).glob(f"data/{_split}-*.parquet"))
            if not matches:
                print(
                    f"error: --split {_split!r} matched no parquet files in "
                    f"{args.repo_id!r}. Available splits can be inferred from the "
                    f"repo's data/ directory listing.",
                    file=sys.stderr,
                )
                return 1

        _local_path = Path(local)
        _size = sum(
            p.stat().st_size for p in _local_path.rglob("*") if p.is_file() and not p.is_symlink()
        )
        if mode == "cache_hit":
            say.ok(
                f"Loaded {_title} from cache in {format_duration(_elapsed)} "
                f"({format_bytes(_size)} on disk)",
                console=_console,
            )
        else:
            say.ok(
                f"Downloaded {_title} in {format_duration(_elapsed)} "
                f"({format_bytes(_size)} on disk)",
                console=_console,
            )
        say.step(f"Location: {local}", console=_console)
        return 0

    if sub == "merge":
        import time as _time

        from .dataset_io import merge_datasets
        from .ui import format_duration, say, status

        say.intent(f"Merging {len(args.inputs)} partials → {args.output}")
        _t0 = _time.perf_counter()
        with status("Validating partials and concatenating"):
            merge_datasets([_Path(p) for p in args.inputs], output_dir=_Path(args.output))
        _elapsed = _time.perf_counter() - _t0
        say.ok(f"Merged in {format_duration(_elapsed)} → {args.output}")
        return 0

    if sub == "info":
        from .dataset_io import read_metadata

        src = args.source
        if _Path(src).exists():
            md = read_metadata(src)
        else:
            from huggingface_hub import hf_hub_download

            md_path = hf_hub_download(
                repo_id=src, filename="metadata.json", repo_type="dataset", revision=args.revision
            )
            md = _json.loads(_Path(md_path).read_text())
        if "splits" in md:
            print("WhestBench dataset (multi-split)")
            print(f"  schema_version: {md['schema_version']}")
            print(f"  format: {md['format']}")
            print(f"  backend: {md['backend']}")
            print(f"  width: {md['width']}  depth: {md['depth']}  n_samples: {md['n_samples']:,}")
            print("  splits:")
            for name in sorted(md["splits"]):
                info = md["splits"][name]
                seed_str = f"  seed={info['seed']}" if "seed" in info else ""
                config_str = f"  config={info['config']}" if "config" in info else ""
                print(f"    {name}:  n_mlps={info['n_mlps']:,}{seed_str}{config_str}")
            print(f"  created_at_utc: {md['created_at_utc']}")
        else:
            print("WhestBench dataset")
            for key in (
                "schema_version",
                "format",
                "backend",
                "seed",
                "split",
                "config",
                "n_mlps",
                "n_samples",
                "width",
                "depth",
                "created_at_utc",
                "device",
                "cuda_device_name",
            ):
                if key in md:
                    print(f"  {key}: {md[key]}")
            proto = md.get("seed_protocol", {})
            if proto:
                print(
                    f"  seed_protocol:  {proto.get('name', '?')} (version {proto.get('version', '?')})"
                )
        return 0

    if sub == "combine-splits":
        from .dataset_io import MergeIncompatibleError, combine_split_datasets

        try:
            out = combine_split_datasets(
                args.input_dirs,
                output_dir=args.output,
                default_split=getattr(args, "default_split", None),
                write_prepared_arrow=not getattr(args, "skip_prepared_arrow", False),
            )
        except (MergeIncompatibleError, FileExistsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        md = _json.loads((_Path(out) / "metadata.json").read_text())
        splits = md.get("splits", {})
        print(f"Combined {len(splits)} splits into {out}:")
        for name in sorted(splits.keys()):
            info = splits[name]
            seed_str = f", seed={info['seed']}" if "seed" in info else ""
            print(f"  - {name}  (n_mlps={info['n_mlps']}{seed_str})")
        if md.get("prepared_splits"):
            print("  prepared Arrow artifacts: " + ", ".join(sorted(md["prepared_splits"].keys())))
        return 0

    if sub == "prepare-arrow":
        from .dataset_io import (
            METADATA_FILE,
            README_FILE,
            build_prepared_splits_for_directory,
            generate_readme,
            read_metadata,
            validate_metadata,
        )

        dataset_dir = _Path(args.dataset_dir)
        if not dataset_dir.is_dir():
            print(f"error: {dataset_dir} is not a directory.", file=sys.stderr)
            return 1
        md = read_metadata(dataset_dir)
        if "splits" not in md:
            print(
                "error: prepare-arrow is only meaningful for multi-split "
                "datasets (no 'splits' block found in metadata.json).",
                file=sys.stderr,
            )
            return 1
        splits = list(md["splits"].keys())
        build_prepared_splits_for_directory(
            dataset_dir,
            splits=splits,
            metadata=md,
        )
        validate_metadata(md)
        (dataset_dir / METADATA_FILE).write_text(_json.dumps(md, indent=2))
        # Re-render README so the YAML frontmatter matches the new metadata.
        ds_size = sum(int(md["splits"][s]["n_mlps"]) for s in splits)
        try:
            from types import SimpleNamespace as _SN

            splits_kwarg = {n: _SN(n_mlps=int(md["splits"][n]["n_mlps"])) for n in splits}
            (dataset_dir / README_FILE).write_text(
                generate_readme(md, splits=splits_kwarg, ds_size=ds_size)
            )
        except Exception as exc:  # noqa: BLE001 — README rewrite is best-effort
            print(
                f"warning: prepared metadata written, but README rewrite "
                f"failed ({exc}). You can re-render via "
                f"`whestbench.dataset_io.generate_readme` separately.",
                file=sys.stderr,
            )
        print(
            f"Patched {dataset_dir} with prepared Arrow artifacts for "
            f"{len(splits)} split(s): "
            f"{', '.join(sorted(md['prepared_splits'].keys()))}"
        )
        return 0

    print("Unknown dataset subcommand. Try --help.", file=sys.stderr)
    return 2


class _RunnerEstimator(BaseEstimator):
    """Adapter that wraps a started runner as a BaseEstimator for scoring."""

    def __init__(self, runner: "Any") -> None:
        self._runner = runner

    def predict(self, mlp: "Any", budget: int) -> fnp.ndarray:
        return self._runner.predict(mlp, budget)

    def last_predict_stats(self) -> Optional[Dict[str, Any]]:
        getter = getattr(self._runner, "last_predict_stats", None)
        if not callable(getter):
            return None
        stats = getter()
        if stats is None:
            return None
        if isinstance(stats, dict):
            return stats
        return {
            "flops_used": getattr(stats, "flops_used", None),
            "wall_time_s": getattr(stats, "wall_time_s", None),
            "flopscope_backend_time_s": getattr(stats, "flopscope_backend_time_s", None),
            "flopscope_overhead_time_s": getattr(stats, "flopscope_overhead_time_s", None),
            "residual_wall_time_s": getattr(stats, "residual_wall_time_s", None),
            "budget_breakdown": getattr(stats, "budget_breakdown", None),
        }


@contextmanager
def _route_scoring_warnings(*, output_format: str) -> Iterator[None]:
    """Route ScoringExhaustionWarnings appropriately for the current output mode.

    - json: suppress entirely (stdout must be pure JSON, stderr must stay quiet).
    - rich/plain: route through rich.get_console().log() so warnings render above
      any active Live region without flicker. Falls back to default formatting
      for non-exhaustion warnings.
    """
    from .scoring import ScoringExhaustionWarning

    if output_format == "json":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ScoringExhaustionWarning)
            yield
        return

    from rich import get_console as _get_console

    console = _get_console()
    original_showwarning = warnings.showwarning

    def _routed(
        message: Any,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: Any = None,
    ) -> None:
        if isinstance(message, ScoringExhaustionWarning):
            console.log(f"[yellow]⚠[/]  {message}")
        else:
            original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = _routed
    try:
        yield
    finally:
        warnings.showwarning = original_showwarning


def _emit_exhaustion_summary(results: Dict[str, Any], *, output_format: str) -> None:
    """If any MLP exhausted budget or time, print a one-line summary to stderr.

    Suppressed for --json mode (caller's responsibility to check).
    """
    per_mlp = results.get("per_mlp") or []
    total = len(per_mlp)
    if total == 0:
        return
    budget_n = sum(
        1 for entry in per_mlp if isinstance(entry, dict) and bool(entry.get("budget_exhausted"))
    )
    time_n = sum(
        1 for entry in per_mlp if isinstance(entry, dict) and bool(entry.get("time_exhausted"))
    )
    if budget_n == 0 and time_n == 0:
        return

    exhausted_total = budget_n + time_n
    parts = []
    if budget_n:
        parts.append(f"{budget_n} FLOP")
    if time_n:
        parts.append(f"{time_n} time")
    breakdown = " and ".join(parts) if parts else ""
    msg = (
        f"{exhausted_total} of {total} MLPs exhausted budget ({breakdown}). "
        f"Pass --fail-fast to stop on first exhaustion; per-MLP tracebacks are "
        f"in the JSON output."
    )

    if output_format == "rich":
        from rich import get_console

        get_console().log(f"[yellow]{msg}[/]")
    else:
        print(msg, file=sys.stderr)


def _run_estimator_with_runner(
    runner: "Any",
    *,
    entrypoint: EstimatorEntrypoint,
    contest_spec: ContestSpec,
    n_mlps: int,
    profile: bool,
    detail: str,
    output_format: str,
    progress: Optional[ProgressCallback] = None,
    contest_data: "Optional[Any]" = None,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """Run estimator through a runner and score against contest data.

    Builds contest data (or accepts a pre-built ``ContestData`` via
    ``contest_data`` when the caller already loaded a dataset), starts the
    runner, wraps it as a BaseEstimator, and delegates scoring to
    ``evaluate_estimator``.
    """
    import time as _time

    spec = contest_spec
    if contest_data is not None:
        data = contest_data
        if progress is not None:
            # Precomputed dataset: emit a single "generating" completion event
            # so progress bars fill immediately rather than staying at zero.
            progress({"phase": "generating", "completed": spec.n_mlps, "total": spec.n_mlps})
    elif progress is not None:
        data = make_contest(
            spec,
            on_sampling_progress=progress,
        )
    else:
        data = make_contest(spec)

    context = SetupContext(
        width=spec.width,
        depth=spec.depth,
        flop_budget=spec.flop_budget,
        api_version="1.0",
        seed=spec.seed if spec.seed is not None else 0,
    )
    limits = ResourceLimits(
        setup_timeout_s=spec.setup_timeout_s,
        predict_timeout_s=spec.predict_timeout_s,
        memory_limit_mb=spec.memory_limit_mb,
        flop_budget=spec.flop_budget,
        wall_time_limit_s=spec.wall_time_limit_s,
        residual_wall_time_limit_s=spec.residual_wall_time_limit_s,
    )

    t0 = _time.time()
    runner.start(entrypoint, context, limits)

    try:
        with _route_scoring_warnings(output_format=output_format):
            results = evaluate_estimator(
                _RunnerEstimator(runner),
                data,
                on_mlp_scored=lambda i: (
                    progress({"phase": "scoring", "completed": i, "total": n_mlps})
                    if progress is not None
                    else None
                ),
                fail_fast=fail_fast,
            )
    finally:
        runner.close()

    elapsed = _time.time() - t0

    if output_format != "json":
        _emit_exhaustion_summary(results, output_format=output_format)

    return {
        "schema_version": "1.1",
        "mode": "human",
        "results": results,
        "run_meta": {
            "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_duration_s": elapsed,
            "host": _host_metadata(),
        },
        "run_config": {
            "n_mlps": n_mlps,
            "width": spec.width,
            "depth": spec.depth,
            "seed": spec.seed,
            "flop_budget": spec.flop_budget,
            "wall_time_limit_s": spec.wall_time_limit_s,
            "residual_wall_time_limit_s": spec.residual_wall_time_limit_s,
        },
    }


def _main_participant(argv: "list[str]") -> int:
    parser = _build_participant_parser()
    if argv and not argv[0].startswith("-"):
        commands = next(
            (
                tuple(action.choices)
                for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)
            ),
            (),
        )
        if argv[0] not in commands:
            matches = difflib.get_close_matches(argv[0], commands, n=1)
            if matches:
                parser.error(f"unknown command '{argv[0]}'. did you mean '{matches[0]}'?")
    args = parser.parse_args(argv)
    command = str(args.command)
    stdout_is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    output_format = resolve_output_format(
        format_arg=getattr(args, "output_format", None),
        json_output=bool(getattr(args, "json_output", False)),
        is_tty=stdout_is_tty,
    )
    if command in {"run", "smoke-test"} and output_format == "rich" and _debugger_active():
        output_format = "plain"
        print(
            "Debugger detected (sys.gettrace / PYTHONBREAKPOINT); forcing plain output.",
            file=sys.stderr,
        )
    json_output = output_format == "json"

    # Apply thread limit early, before any backend module is imported.
    max_threads = getattr(args, "max_threads", None)
    if max_threads is not None:
        from .concurrency import apply_thread_limit

        apply_thread_limit(max_threads)
    debug = bool(getattr(args, "debug", False))
    no_rich = output_format == "plain"

    try:
        if command == "smoke-test":
            import time as _time

            from .ui import format_duration, say

            if json_output:
                report = run_default_report(
                    profile=bool(args.profile),
                    detail=str(args.detail),
                )
                report["mode"] = "agent"
                print(render_agent_report(_json_payload_with_metadata(report)), end="")
                return 0

            say.intent("Running smoke test against CombinedEstimator")
            _smoke_t0 = _time.perf_counter()

            if no_rich:
                # Plain-text path: no Rich Live, no progress bar. Emit
                # single-line phase updates to stderr so a log-scraper still
                # gets progress signal without terminal control sequences.
                def _plain_smoke_progress(event: Dict[str, Any]) -> None:
                    phase = str(event.get("phase", ""))
                    completed = int(event.get("completed", 0))
                    total = int(event.get("total", 0) or 0)
                    if total:
                        print(f"[smoke-test] {phase}: {completed}/{total}", file=sys.stderr)
                    else:
                        print(f"[smoke-test] {phase}: {completed}", file=sys.stderr)

                report = run_default_report(
                    profile=bool(args.profile),
                    detail=str(args.detail),
                    progress=_plain_smoke_progress,
                )
                report["mode"] = "human"
                output = _render_plain_text_report(report, command="smoke-test")
                print(output, end="" if output.endswith("\n") else "\n")
                say.ok(
                    f"Smoke test completed in {format_duration(_time.perf_counter() - _smoke_t0)}"
                )
                return 0

            # Set up Rich progress bar for immediate user feedback.
            try:
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    TimeElapsedColumn,
                )

                spec = _smoke_test_contest_spec()
                progress_bar = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=None),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                )
                gt_task = progress_bar.add_task("Computing ground truth", total=spec.n_mlps)
                score_task = progress_bar.add_task(
                    "Scoring estimator", total=spec.n_mlps, visible=False
                )

                def _on_smoke_progress(event: Dict[str, Any]) -> None:
                    phase = str(event.get("phase", ""))
                    completed = int(event.get("completed", 0))
                    if phase == "ground_truth":
                        progress_bar.update(gt_task, completed=completed)
                    elif phase == "scoring":
                        progress_bar.update(score_task, visible=True)
                        progress_bar.update(score_task, completed=completed)

                with progress_bar:
                    report = run_default_report(
                        profile=bool(args.profile),
                        detail=str(args.detail),
                        progress=_on_smoke_progress,
                    )
            except ImportError:
                report = run_default_report(profile=bool(args.profile), detail=str(args.detail))

            report["mode"] = "human"
            rich_rendered = False
            smoke_doc = build_smoke_test_presentation(report, debug=debug)
            try:
                output = render_human_report(
                    report,
                    show_diagnostic_plots=bool(args.show_diagnostic_plots),
                    debug=debug,
                    presentation_doc=replace(
                        smoke_doc,
                        sections=[
                            section
                            for section in smoke_doc.sections
                            if not (
                                isinstance(section, StepsSection) and section.title == "Next Steps"
                            )
                        ],
                        epilogue_messages=[],
                    ),
                )
                rich_rendered = True
            except Exception as exc:
                print(
                    f"Rich dashboard unavailable ({exc}); falling back to plain-text report.",
                    file=sys.stderr,
                )
                output = _render_plain_text_report(report, command="smoke-test")
            print(output, end="" if output.endswith("\n") else "\n")
            if rich_rendered:
                next_steps = render_smoke_test_next_steps(report, debug=debug)
                print(next_steps, end="" if next_steps.endswith("\n") else "\n")
            say.ok(f"Smoke test completed in {format_duration(_time.perf_counter() - _smoke_t0)}")
            return 0

        if command == "version":
            payload = {
                "ok": True,
                "command": "version",
                "name": "whestbench",
                "version": _resolve_whestbench_version(),
            }
            if json_output:
                print(json.dumps(_json_payload_with_metadata(payload), indent=2))
            else:
                print(f"whestbench {payload['version']}")
            return 0

        if command == "init":
            import time as _time

            from .ui import format_duration, say

            _target = Path(args.path).resolve()
            say.intent(
                f"Initializing starter estimator in {_target}",
                quiet=json_output,
            )
            _t0 = _time.perf_counter()
            created = _write_init_template(_target)
            _elapsed = _time.perf_counter() - _t0
            payload = {"ok": True, "created": created}
            if json_output:
                print(json.dumps(_json_payload_with_metadata(payload), indent=2))
            else:
                doc = build_init_presentation(payload)
                print(
                    render_command_presentation(
                        doc,
                        output_format=output_format,
                        force_terminal=stdout_is_tty,
                    ),
                    end="",
                )
            _n = len(created)
            if _n:
                _plural = "s" if _n != 1 else ""
                say.ok(
                    f"Created {_n} file{_plural} in {format_duration(_elapsed)}",
                    quiet=json_output,
                )
            else:
                say.ok(
                    f"Starter files already present (checked in {format_duration(_elapsed)})",
                    quiet=json_output,
                )
            return 0

        if command == "validate":
            import time as _time

            from .ui import format_duration, say, status

            validate_seed: Optional[int] = getattr(args, "seed", None)
            say.intent(
                f"Validating estimator {args.estimator}",
                quiet=json_output,
            )
            _t0 = _time.perf_counter()
            if json_output:
                payload = _json_payload_with_metadata(
                    validate_submission_entrypoint(
                        args.estimator, class_name=args.class_name, seed=validate_seed
                    )
                )
                print(json.dumps(payload, indent=2))
                return 0

            with status(
                f"Importing {args.estimator} and running setup/predict checks",
                quiet=json_output,
            ):
                result = _run_validate_checks(
                    args.estimator, class_name=args.class_name, seed=validate_seed
                )
                _elapsed = _time.perf_counter() - _t0
                doc = build_validate_presentation(result)
                print(
                    render_command_presentation(
                        doc,
                        output_format=output_format,
                        force_terminal=stdout_is_tty,
                    ),
                    end="",
                )
            say.ok(
                f"Validation passed in {format_duration(_elapsed)}",
                quiet=json_output,
            )
            return 0

        if command == "create-dataset":
            print(
                "ERROR: `whest create-dataset` has been replaced by `whest dataset bake`.\n"
                "  See: whest dataset bake --help\n"
                "  Migration: same args but --output-path FILE.npz becomes --output DIR/",
                file=sys.stderr,
            )
            return 2

        if command == "dataset":
            return _dispatch_dataset_command(args)

        if command == "run":
            normalized_runner = _normalize_runner_name(str(args.runner))
            runner = LocalRunner() if normalized_runner == "local" else SubprocessRunner()
            contest_spec = _default_contest_spec()
            user_n_mlps: Optional[int] = int(args.n_mlps) if args.n_mlps is not None else None
            run_seed: Optional[int] = getattr(args, "seed", None)
            if user_n_mlps is not None and user_n_mlps <= 0:
                raise ValueError("--n-mlps must be positive.")
            flop_budget = (
                int(args.flop_budget) if args.flop_budget is not None else contest_spec.flop_budget
            )
            gt_samples = (
                int(args.n_samples)
                if args.n_samples is not None
                else contest_spec.ground_truth_samples
            )
            entrypoint = EstimatorEntrypoint(
                file_path=Path(args.estimator).resolve(),
                class_name=args.class_name,
            )

            # --- Dataset loading & n_mlps resolution ---
            dataset_path: Optional[str] = getattr(args, "dataset", None)
            contest_data = None
            ds_meta: Dict[str, Any] = {}
            # `rev` / `_is_local` are populated inside the
            # `if dataset_path is not None:` branch below; pre-init so pyright
            # sees them as always-bound when referenced again in the later
            # report-building blocks (which also guard on `dataset_path`).
            rev: Optional[str] = None
            _is_local: bool = True
            if dataset_path is not None:
                import time as _time

                from rich.console import Console as _RichConsole

                from .dataset import load_dataset as _wb_load_dataset
                from .hf_progress import hf_download, hf_preflight
                from .scoring import make_contest_from_dataset
                from .ui import format_bytes, format_duration, say

                repo_or_path, rev, _is_local = _resolve_dataset_arg(
                    dataset_path, revision=getattr(args, "revision", None)
                )
                _streaming = bool(getattr(args, "streaming", False))

                if _is_local and _streaming:
                    print(
                        "error: --streaming is only valid with hf:// datasets, not local paths.",
                        file=sys.stderr,
                    )
                    return 1

                if _is_local:
                    ds = _wb_load_dataset(
                        repo_or_path, revision=rev, split=getattr(args, "split", None)
                    )
                else:
                    _console = _RichConsole()
                    _title = f"hf://{repo_or_path}@{rev or 'main'}"
                    say.step(f"Resolving {_title}", console=_console, quiet=json_output)

                    # Resolve --split (or its default_split fallback) BEFORE
                    # preflight so the size estimate + the parquet download
                    # touch only the target split. Without this, a multi-split
                    # dataset with default_split declared but --split omitted
                    # would preflight every shard in the repo (~4.7 GB across
                    # mini+full instead of ~250 MB for mini alone).
                    _user_split = getattr(args, "split", None)
                    _effective_split = _user_split
                    # `_md_early` is populated from a one-shot metadata.json
                    # fetch when we need to resolve `default_split` OR detect
                    # `prepared_splits` for the preflight sizing path.
                    _md_early: "Dict[str, Any] | None" = None
                    try:
                        from huggingface_hub import hf_hub_download as _hf_hub_dl

                        _md_path = _hf_hub_dl(
                            repo_id=repo_or_path,
                            repo_type="dataset",
                            revision=rev,
                            filename="metadata.json",
                        )
                        _md_early = json.loads(Path(_md_path).read_text())
                    except Exception:  # noqa: BLE001 — best-effort optimisation
                        # If we can't fetch metadata.json early (network blip,
                        # gated repo, etc.) we fall through; preflight + load
                        # handle the error path normally.
                        _md_early = None

                    if _user_split is None and _md_early is not None and "splits" in _md_early:
                        _ds_default = _md_early.get("default_split")
                        if isinstance(_ds_default, str) and _ds_default in _md_early["splits"]:
                            _effective_split = _ds_default
                            if not json_output:
                                say.ok(
                                    f"Using default split "
                                    f"{_ds_default!r} (from "
                                    f"metadata.default_split)",
                                    console=_console,
                                )

                    # If this dataset ships a prepared-Arrow subtree for the
                    # effective split, point preflight at that subtree so the
                    # advertised "Downloading N files, X bytes" matches what
                    # `whestbench.load_dataset` will actually pull.
                    _prepared_prefix: "str | None" = None
                    if (
                        _effective_split is not None
                        and _md_early is not None
                        and isinstance(_md_early.get("prepared_splits"), dict)
                    ):
                        _prep_entry = _md_early["prepared_splits"].get(_effective_split)
                        if isinstance(_prep_entry, dict):
                            _prep_path = _prep_entry.get("path")
                            if isinstance(_prep_path, str) and _prep_path:
                                # Ensure a trailing slash so prefix matching is
                                # directory-scoped (e.g. "prepared/mini/" — NOT
                                # "prepared/min" which would match a hypothetical
                                # "prepared/minimal/").
                                _prepared_prefix = _prep_path.rstrip("/") + "/"

                    preflight = hf_preflight(
                        repo_or_path,
                        revision=rev,
                        split=_effective_split,
                        data_subtree_prefix=_prepared_prefix,
                    )

                    if _streaming:
                        mode: Literal["cache_hit", "materialize", "streaming"] = "streaming"
                    elif preflight is not None and preflight.is_cached:
                        mode = "cache_hit"
                    else:
                        mode = "materialize"

                    if _streaming:
                        say.warn(
                            "Streaming from HF\n"
                            "  • Iteration-only: streaming yields MLPs sequentially "
                            "without random access.\n"
                            "  • Not cached locally — every run re-fetches from "
                            "the network.\n"
                            "  • Best for quick iteration with `--n-mlps K` where "
                            "K is small.\n"
                            "  • To populate the cache, run:\n"
                            f"      whest dataset download {repo_or_path} "
                            f"--revision {rev or 'main'}",
                            console=_console,
                            quiet=json_output,
                        )
                    elif mode == "materialize":
                        if preflight is not None:
                            _files = preflight.file_count
                            _plural = "s" if _files != 1 else ""
                            say.intent(
                                f"Downloading {_title} — {_files} file{_plural}, "
                                f"{format_bytes(preflight.total_bytes)}",
                                console=_console,
                                quiet=json_output,
                            )
                        else:
                            say.intent(
                                f"Downloading {_title} (preflight unavailable)",
                                console=_console,
                                quiet=json_output,
                            )

                    _t0 = _time.perf_counter()
                    with hf_download(
                        _console,
                        title=_title,
                        preflight=preflight,
                        mode=mode,
                        quiet=json_output,
                    ):
                        ds = _wb_load_dataset(
                            repo_or_path,
                            revision=rev,
                            split=_effective_split,
                            streaming=_streaming,
                        )
                    _elapsed = _time.perf_counter() - _t0

                    from datasets import (
                        Dataset as _Dataset,
                    )

                    if isinstance(ds, _Dataset):
                        _n_str = f"{len(ds):,}"
                    else:
                        # IterableDataset (streaming) or *Dict types — no random
                        # access / __len__. The *Dict cases are rejected below
                        # with a clearer error.
                        _n_str = "?"
                    if mode == "cache_hit":
                        say.ok(
                            f"Loaded {_n_str} MLPs in {format_duration(_elapsed)} (from cache)",
                            console=_console,
                            quiet=json_output,
                        )
                    elif mode == "streaming":
                        say.ok(
                            f"Streaming dataset ready in {format_duration(_elapsed)}",
                            console=_console,
                            quiet=json_output,
                        )
                    else:
                        _bytes_label = (
                            f" {format_bytes(preflight.total_bytes)}" if preflight else ""
                        )
                        say.ok(
                            f"Downloaded{_bytes_label} and loaded {_n_str} MLPs in {format_duration(_elapsed)}",
                            console=_console,
                            quiet=json_output,
                        )
                from datasets import DatasetDict as _DatasetDict
                from datasets import IterableDataset as _IterableDataset
                from datasets import IterableDatasetDict as _IterableDatasetDict

                if isinstance(ds, (_DatasetDict, _IterableDatasetDict)):
                    # --split was not given. If the dataset's metadata declares
                    # a `default_split`, project the dict down to that single
                    # split; otherwise instruct the user to pass --split.
                    _dd_meta = _wb_metadata(ds)
                    _default_split = _dd_meta.get("default_split")
                    if isinstance(_default_split, str) and _default_split in ds:
                        if not json_output:
                            print(
                                f"Using default split {_default_split!r} "
                                f"(from metadata.default_split)",
                                file=sys.stderr,
                            )
                        ds = ds[_default_split]
                    else:
                        _hint = (
                            ""
                            if _default_split is None
                            else (
                                f" (metadata's default_split="
                                f"{_default_split!r} is not one of the "
                                f"available splits and is being ignored)"
                            )
                        )
                        print(
                            f"error: dataset {dataset_path!r} is multi-split "
                            f"with splits {sorted(ds.keys())}. Pass --split "
                            f"<name> to select one{_hint}.",
                            file=sys.stderr,
                        )
                        return 1
                ds_meta = _wb_metadata(ds)

                if isinstance(ds, _IterableDataset):
                    ds_n_mlps = int(ds_meta.get("n_mlps") or 0)
                    if ds_n_mlps <= 0:
                        raise SystemExit("error: streaming dataset has no n_mlps in metadata.")
                else:
                    ds_n_mlps = len(ds)

                if user_n_mlps is None:
                    n_mlps = ds_n_mlps
                elif user_n_mlps > ds_n_mlps:
                    print(
                        f"Warning: --n-mlps={user_n_mlps} exceeds dataset size "
                        f"({ds_n_mlps}); using {ds_n_mlps}.",
                        file=sys.stderr,
                    )
                    n_mlps = ds_n_mlps
                else:
                    n_mlps = user_n_mlps

                contest_spec = ContestSpec(
                    width=ds_meta["width"],
                    depth=ds_meta["depth"],
                    n_mlps=n_mlps,
                    flop_budget=flop_budget,
                    ground_truth_samples=gt_samples,
                    seed=run_seed,
                    wall_time_limit_s=getattr(args, "wall_time_limit", None),
                    residual_wall_time_limit_s=getattr(args, "residual_wall_time_limit", None),
                )
                contest_data = make_contest_from_dataset(contest_spec, ds, n_mlps)
            else:
                n_mlps = user_n_mlps if user_n_mlps is not None else 10
                contest_spec = ContestSpec(
                    width=contest_spec.width,
                    depth=contest_spec.depth,
                    n_mlps=n_mlps,
                    flop_budget=flop_budget,
                    ground_truth_samples=gt_samples,
                    seed=run_seed,
                    wall_time_limit_s=getattr(args, "wall_time_limit", None),
                    residual_wall_time_limit_s=getattr(args, "residual_wall_time_limit", None),
                )

            score_kwargs: Dict[str, Any] = {
                "entrypoint": entrypoint,
                "contest_spec": contest_spec,
                "n_mlps": n_mlps,
                "profile": bool(args.profile),
                "detail": str(args.detail),
                "output_format": output_format,
                "contest_data": contest_data,
                "fail_fast": bool(getattr(args, "fail_fast", False)),
            }

            used_plain_fallback = False
            _tip_console = Console(highlight=False)
            _dataset_tip = ""
            if json_output:
                report = _run_estimator_with_runner(runner, **score_kwargs)
                report["mode"] = "agent"
                if dataset_path is not None:
                    # `dataset_path` is the raw user argument — could be a
                    # local directory OR an hf:// URL OR a bare owner/repo.
                    # Only `resolve()` it for local paths; HF Hub URLs are
                    # carried through verbatim so the report stays interpretable.
                    report.setdefault("run_config", {})["dataset"] = {
                        "path": str(Path(dataset_path).resolve()) if _is_local else dataset_path,
                        "sha256": _metadata_file_hash(dataset_path, revision=rev),
                        "seed": ds_meta.get("seed"),
                        "n_mlps": ds_meta.get("n_mlps"),
                    }
                output = render_agent_report(_json_payload_with_metadata(report))
            else:
                metadata = resolve_estimator_class_metadata(
                    entrypoint.file_path, class_name=entrypoint.class_name
                )
                pre_report = _pre_run_report(
                    n_mlps=n_mlps,
                    contest_spec=contest_spec,
                    profile=bool(args.profile),
                    detail=str(args.detail),
                    estimator_class=metadata.class_name,
                    estimator_path=args.estimator,
                )
                total_units = (
                    n_mlps
                    if dataset_path is not None
                    else n_mlps
                    * sample_layer_statistics_chunk_count(
                        contest_spec.width, contest_spec.ground_truth_samples
                    )
                )
                _dataset_tip = (
                    "\n[bold bright_yellow]Tip:[/] Ground truth is recomputed on every run. "
                    "Consider baking and reusing a dataset:\n"
                    "   [cyan]whest dataset bake[/] [green]--n-mlps[/] [yellow]10[/] [green]--n-samples[/] [yellow]10000[/] [green]--width[/] [yellow]256[/] [green]--depth[/] [yellow]8[/] [green]--output[/] [yellow]./my-eval[/]\n"
                    "   [cyan]whest run[/] [green]--estimator[/] [yellow]...[/] [green]--dataset[/] [yellow]./my-eval[/]\n"
                )
                gen_label = (
                    "Loading dataset" if dataset_path is not None else "Sampling Ground Truth"
                )
                if no_rich:
                    # Plain-text path: skip every Rich Live/progress display so
                    # breakpoints in predict() reach a clean terminal. Emit
                    # single-line phase updates to stderr so logs still show
                    # progress without terminal control sequences.
                    print(f"[run] estimator_class={metadata.class_name}", file=sys.stderr)
                    print(f"[run] estimator_path={args.estimator}", file=sys.stderr)
                    print(f"[run] n_mlps={n_mlps} gen_label={gen_label}", file=sys.stderr)
                    score_kwargs["progress"] = _PlainRunProgressLogger(n_mlps=n_mlps)
                    report = _run_estimator_with_runner(runner, **score_kwargs)
                elif rich_tqdm is None:
                    _print_human_startup(
                        pre_report,
                        estimator_class=metadata.class_name,
                        estimator_path=args.estimator,
                    )

                    with _progress_callback(
                        total_units, n_mlps, gen_label=gen_label
                    ) as progress_cb:
                        score_kwargs["progress"] = progress_cb
                        report = _run_estimator_with_runner(runner, **score_kwargs)
                else:
                    _print_human_startup(
                        pre_report,
                        estimator_class=metadata.class_name,
                        estimator_path=args.estimator,
                    )

                    with _live_top_pane_session(
                        pre_report, total_units, n_mlps, gen_label=gen_label
                    ) as live_session:
                        score_kwargs["progress"] = live_session.on_progress
                        report = _run_estimator_with_runner(runner, **score_kwargs)
                        live_session.update_run_meta(report.get("run_meta", {}))
                report["mode"] = "human"
                if dataset_path is not None:
                    # `dataset_path` is the raw user argument — could be a
                    # local directory OR an hf:// URL OR a bare owner/repo.
                    # Only `resolve()` it for local paths; HF Hub URLs are
                    # carried through verbatim so the report stays interpretable.
                    report.setdefault("run_config", {})["dataset"] = {
                        "path": str(Path(dataset_path).resolve()) if _is_local else dataset_path,
                        "sha256": _metadata_file_hash(dataset_path, revision=rev),
                        "seed": ds_meta.get("seed"),
                        "n_mlps": ds_meta.get("n_mlps"),
                    }
                report = _merge_pre_run_context(report, pre_report)
                if no_rich:
                    output = _render_plain_text_report(
                        report,
                        debug=debug,
                        include_epilogues=False,
                    )
                else:
                    try:
                        output = render_human_results(
                            report,
                            show_diagnostic_plots=bool(args.show_diagnostic_plots),
                            debug=debug,
                        )
                    except Exception as exc:
                        print(
                            f"Rich dashboard unavailable ({exc}); falling back to plain-text report.",
                            file=sys.stderr,
                        )
                        output = _render_plain_text_report(
                            report,
                            debug=debug,
                            include_diagnostic_plots_tip=not bool(args.show_diagnostic_plots),
                            include_context=False,
                        )
                        used_plain_fallback = True
            print(output, end="" if output.endswith("\n") else "\n")
            if not json_output and not no_rich:
                if not used_plain_fallback:
                    _tip_console.print(
                        "[bold bright_yellow]Tip:[/] Use [green]--format json[/] for JSON output when calling from automated agents or UIs."
                    )
                    if not args.show_diagnostic_plots:
                        _tip_console.print(
                            "[bold bright_yellow]Tip:[/] Use [green]--show-diagnostic-plots[/] to include diagnostic plot panes."
                        )
                if dataset_path is None:
                    _tip_console.print(_dataset_tip)
            per_mlp = report.get("results", {}).get("per_mlp", [])
            failing = [e for e in per_mlp if isinstance(e, dict) and e.get("error")]
            if failing:
                hint = "rerun with --debug for tracebacks" if not debug else "see tracebacks above"
                print(
                    f"{len(failing)} of {len(per_mlp)} MLP(s) raised during predict; "
                    f"{hint}. Use --fail-fast to stop on first error.",
                    file=sys.stderr,
                )
                return 1
            return 0

        if command == "package":
            import time as _time

            from .ui import format_bytes, format_duration, progress_bytes, say

            # The bytes flowing into the bar are gzipped output bytes; counting
            # uncompressed input bytes as the bar's target gives a roughly
            # honest progress signal (compressed size is typically ≤ input
            # size + per-file tar headers + manifest blob). A small overshoot
            # is benign — the bar still conveys real-time motion. We add a
            # ~1 KB pad for the manifest + tar overhead so the bar reaches
            # close to 100% rather than sitting at "1.5x" by the time the
            # last file is added.
            _estimator = Path(args.estimator)
            _input_paths: list[Path] = [_estimator] if _estimator.is_file() else []
            for _opt in (args.requirements, args.submission_metadata, args.approach):
                if _opt is None:
                    continue
                _p = Path(_opt)
                if _p.is_file():
                    _input_paths.append(_p)
            _total = sum(p.stat().st_size for p in _input_paths) + 1024

            say.intent(
                f"Packaging {args.estimator} → {args.output or 'submission-*.tar.gz'}",
                quiet=json_output,
            )
            _t0 = _time.perf_counter()
            with progress_bytes(total=_total, label="Packaging", quiet=json_output) as _bar:
                artifact_path = package_submission(
                    args.estimator,
                    class_name=args.class_name,
                    requirements_path=args.requirements,
                    submission_yaml_path=args.submission_metadata,
                    approach_md_path=args.approach,
                    output_path=args.output,
                    progress=lambda n: _bar.advance(n),
                )
            _elapsed = _time.perf_counter() - _t0
            try:
                _size = artifact_path.stat().st_size
                _size_label = f" ({format_bytes(_size)})"
            except OSError:
                _size_label = ""
            say.ok(
                f"Wrote {artifact_path} in {format_duration(_elapsed)}{_size_label}",
                quiet=json_output,
            )

            payload = {"ok": True, "artifact_path": str(artifact_path)}
            if json_output:
                print(json.dumps(_json_payload_with_metadata(payload), indent=2))
            else:
                doc = build_package_presentation(payload)
                print(
                    render_command_presentation(
                        doc,
                        output_format=output_format,
                        force_terminal=stdout_is_tty,
                    ),
                    end="",
                )
            return 0

        if command == "doctor":
            import time as _time

            from .doctor import _doctor_exit_code, run_all
            from .reporting import render_doctor_json, render_doctor_report
            from .ui import format_duration, say, status

            say.intent("Running whestbench doctor", quiet=json_output)
            _t0 = _time.perf_counter()
            with status(
                "Probing Python, uv, install mode, BLAS, disk, and CWD permissions",
                quiet=json_output,
            ):
                checks = run_all(debug=debug)
            _elapsed = _time.perf_counter() - _t0
            if output_format == "json":
                print(
                    json.dumps(
                        _json_payload_with_metadata(json.loads(render_doctor_json(checks))),
                        indent=2,
                    )
                )
            elif output_format == "plain":
                print(render_doctor_report(checks, rich=False), end="")
            else:
                print(render_doctor_report(checks, rich=True), end="")
            _n = len(checks)
            _plural = "s" if _n != 1 else ""
            say.ok(
                f"Ran {_n} check{_plural} in {format_duration(_elapsed)}",
                quiet=json_output,
            )
            return _doctor_exit_code(checks, strict=bool(args.strict))

        if command == "profile-simulation":
            import time as _time

            from .profiler import run_profile
            from .ui import format_duration, say

            say.intent(
                f"Profiling simulation backends (preset={args.preset})",
                quiet=json_output,
            )
            _t0 = _time.perf_counter()
            terminal_output, json_data = run_profile(
                preset_name=str(args.preset),
                output_path=args.output,
                show_progress=output_format == "rich",
                max_threads=args.max_threads,
                verbose=bool(args.verbose),
                log_progress=bool(args.log_progress),
                output_format=output_format,
            )
            _elapsed = _time.perf_counter() - _t0
            if json_output:
                print(json.dumps(_json_payload_with_metadata(json_data or {}), indent=2))
            else:
                print(terminal_output, end="" if terminal_output.endswith("\n") else "\n")
            say.ok(
                f"Profile completed in {format_duration(_elapsed)}",
                quiet=json_output,
            )
            return 0

        if command == "login":
            import getpass

            from . import aicrowd_config as _cfg
            from .ui import say

            api_key = args.api_key or os.environ.get("AICROWD_API_KEY")
            if not api_key:
                if json_output:
                    print(json.dumps({"ok": False, "error": "no api key provided"}))
                    return 2
                api_key = getpass.getpass("AIcrowd API key: ").strip()
            if not api_key:
                say.warn("No API key entered; aborting.")
                return 2

            say.intent("Verifying AIcrowd API key", quiet=json_output)
            try:
                ident = _aicrowd_verify_identity(api_key)
            except Exception as e:  # AIcrowdAPIError or transport error
                if json_output:
                    print(json.dumps({"ok": False, "error": str(e)}))
                else:
                    say.warn(f"Login failed: {e}")
                    say.hint("Copy your key from your AIcrowd profile page and try again.")
                return 1

            path = _cfg.save_api_key(api_key)
            name = ident.get("username") or ident.get("id")
            if json_output:
                print(json.dumps({"ok": True, "identity": ident, "config_path": str(path)}))
            else:
                say.ok(f"Logged in as {name}")
                say.hint(f"Key saved to {path}")
            return 0

        if command == "submit":
            import time as _time

            from . import aicrowd_config as _cfg
            from .aicrowd_client import extract_submission_id
            from .ui import say

            try:
                api_key = _cfg.resolve_api_key(args.api_key)
            except _cfg.NotLoggedIn as e:
                if json_output:
                    print(json.dumps({"ok": False, "error": str(e)}))
                else:
                    say.warn(str(e))
                    say.hint("Run `whest login` first.")
                return 2

            # Resolve the artifact: package an estimator if --estimator was given.
            if args.estimator:
                say.intent(f"Packaging {args.estimator}", quiet=json_output)
                artifact = str(
                    package_submission(
                        args.estimator,
                        class_name=args.class_name,
                        requirements_path=args.requirements,
                        submission_yaml_path=args.submission_metadata,
                        approach_md_path=args.approach,
                    )
                )
            else:
                artifact = args.artifact
            if not Path(artifact).is_file():
                msg = f"submission artifact not found: {artifact}"
                if json_output:
                    print(json.dumps({"ok": False, "error": msg}))
                else:
                    say.warn(msg)
                return 2

            client = AIcrowdClient(api_key=api_key)
            try:
                say.intent("Submitting to AIcrowd", quiet=json_output)
                pid = client.verify_identity()
                challenge_id = client.resolve_challenge(args.challenge)
                if not client.check_registration(challenge_id=challenge_id, participant_id=pid):
                    msg = f"You are not registered for '{args.challenge}'."
                    if json_output:
                        print(json.dumps({"ok": False, "error": msg}))
                    else:
                        say.warn(msg)
                        say.hint(
                            "Accept the challenge rules at "
                            f"https://www.aicrowd.com/challenges/{args.challenge}"
                        )
                    return 1
                upload = client.get_upload_details(challenge_slug=args.challenge)
                say.step("Uploading artifact to S3", quiet=json_output)
                s3_key = client.upload_to_s3(upload=upload, file_path=artifact)
                sub = client.create_submission(
                    challenge_slug=args.challenge, s3_key=s3_key, description=args.description
                )
            except Exception as e:
                if json_output:
                    print(json.dumps({"ok": False, "error": str(e)}))
                else:
                    say.warn(f"Submission failed: {e}")
                return 1

            sub_id = extract_submission_id(sub)
            if not json_output:
                say.ok(f"Submitted (submission id {sub_id})")
                say.hint(
                    "Track it at "
                    f"https://www.aicrowd.com/challenges/{args.challenge}/submissions/{sub_id}"
                )

            final = sub
            if args.watch and sub_id is not None:
                # Best-effort poll. The submission is already created+grading
                # asynchronously on AIcrowd; status polling is a convenience and
                # must never turn a successful submit into a failure. (Some
                # deployments don't expose a participant-facing single-submission
                # status endpoint — degrade gracefully rather than crash.)
                say.intent("Waiting for grading", quiet=json_output)
                terminal = {"graded", "failed"}
                try:
                    while str(final.get("grading_status_cd")) not in terminal:
                        _time.sleep(5.0)
                        final = client.get_submission_status(int(sub_id))
                        say.step(f"status: {final.get('grading_status_cd')}", quiet=json_output)
                    status = str(final.get("grading_status_cd"))
                    if not json_output:
                        if status == "graded":
                            say.ok(f"Graded — score {final.get('score')}")
                        else:
                            say.warn(f"Grading {status}: {final.get('grading_message', '')}")
                except Exception as e:  # noqa: BLE001 - watch is best-effort
                    if not json_output:
                        say.warn(f"Couldn't poll grading status ({e}).")
                        say.hint(
                            "The submission was created — track it at "
                            f"https://www.aicrowd.com/challenges/{args.challenge}/submissions/{sub_id}"
                        )

            if json_output:
                print(json.dumps({"ok": True, "submission_id": sub_id, "submission": final}))
            return 0 if str(final.get("grading_status_cd", "submitted")) != "failed" else 1

        raise ValueError(f"Unsupported command: {command}")
    except Exception as exc:  # pragma: no cover - exercised by CLI tests
        stage = exc.stage if isinstance(exc, RunnerError) else command
        payload = _error_payload(exc, include_traceback=debug, stage=stage)
        _print_error(
            payload,
            json_output=json_output,
            debug=debug,
            output_format="rich" if output_format == "rich" else "plain",
            force_terminal=stdout_is_tty,
            show_inprocess_hint=(
                command == "run"
                and _normalize_runner_name(str(getattr(args, "runner", "local"))) == "subprocess"
            ),
        )
        return 1


def main(argv: "Optional[list[str]]" = None) -> int:
    """Dispatch participant subcommands."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    return _main_participant(args_list)


def _print_error(
    payload: Dict[str, Any],
    *,
    json_output: bool,
    debug: bool,
    output_format: Literal["rich", "plain"] = "plain",
    force_terminal: bool = False,
    show_inprocess_hint: bool = False,
) -> None:
    if json_output:
        print(json.dumps(_json_payload_with_metadata(payload), indent=2))
        return
    doc = build_error_presentation(
        payload,
        debug=debug,
        show_inprocess_hint=show_inprocess_hint,
    )
    print(
        render_command_presentation(
            doc,
            output_format=output_format,
            force_terminal=force_terminal,
        ),
        end="",
    )


def _extract_exception_details(exc: Exception) -> Dict[str, Any] | None:
    details = exc.detail.details if isinstance(exc, RunnerError) else getattr(exc, "details", None)
    if isinstance(details, dict) and details:
        return details
    return None


def _error_payload(
    exc: Exception,
    *,
    include_traceback: bool,
    stage: str = "scoring",
) -> Dict[str, Any]:
    """Build stable error payload shape for human/JSON mode outputs."""
    message = str(exc) or exc.__class__.__name__
    error: Dict[str, Any] = {
        "stage": stage,
        "code": _error_code(exc, message),
        "message": message,
    }
    details = _extract_exception_details(exc)
    if details is not None:
        error["details"] = details
    if include_traceback:
        error["traceback"] = traceback.format_exc()
    return {"ok": False, "error": error}


def _error_code(exc: Exception, message: str) -> str:
    """Map common failures to stable error codes."""
    if isinstance(exc, RunnerError):
        return exc.detail.code
    lowered = message.lower()
    if isinstance(exc, ValueError):
        if "must have shape" in lowered:
            return "ESTIMATOR_BAD_SHAPE"
        if "finite" in lowered:
            return "ESTIMATOR_NON_FINITE"
        return "SCORING_VALIDATION_ERROR"
    return "SCORING_RUNTIME_ERROR"
