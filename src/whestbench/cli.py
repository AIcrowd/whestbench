"""CLI and convenience entrypoints for local scoring runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Literal, Optional, overload

import whest as we
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
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

from .estimators import CombinedEstimator
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint
from .loader import load_estimator_from_path, resolve_estimator_class_metadata
from .packaging import package_submission
from .presentation.adapters import (
    build_create_dataset_presentation,
    build_error_presentation,
    build_init_presentation,
    build_package_presentation,
    build_run_presentation,
    build_smoke_test_presentation,
    build_validate_presentation,
)
from .presentation.render_plain import render_plain_presentation
from .presentation.render_rich import render_rich_presentation
from .reporting import (
    _compute_gauge_state,
    _fmt_flops,
    _gauge_bar_fragment,
    _select_top_over_budget,
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

_DEFAULT_ESTIMATOR = CombinedEstimator()
ProgressCallback = Callable[[Dict[str, Any]], None]


def _default_contest_spec() -> ContestSpec:
    return ContestSpec(
        width=100,
        depth=16,
        n_mlps=10,
        flop_budget=100_000_000,
        ground_truth_samples=100 * 100 * 256,
    )


def _default_resource_limits() -> ResourceLimits:
    return ResourceLimits(
        setup_timeout_s=5.0,
        predict_timeout_s=30.0,
        memory_limit_mb=4096,
        flop_budget=100_000_000,
        cpu_time_limit_s=None,
    )


def _debugger_active() -> bool:
    """Detect whether the user has opted into a debugger workflow.

    Returns True if any of:
    - ``sys.gettrace()`` returns a non-None trace function (e.g. running
      under ``python -m pdb`` or an attached debugger).
    - ``PYTHONBREAKPOINT`` is set to a non-empty, non-"0" value (CPython's
      standard env var that controls ``breakpoint()``).
    - ``WHESTBENCH_NO_RICH`` is set to a non-empty, non-"0" value
      (project-specific escape hatch for CI / non-TTY logs).
    """
    if sys.gettrace() is not None:
        return True
    pb = os.environ.get("PYTHONBREAKPOINT", "").strip()
    if pb and pb != "0":
        return True
    no_rich_env = os.environ.get("WHESTBENCH_NO_RICH", "").strip()
    if no_rich_env and no_rich_env != "0":
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
    score = result["primary_score"]
    if profile:
        return score, list(result.get("per_mlp", []))
    return score


def _smoke_test_contest_spec() -> ContestSpec:
    """Lightweight spec for the smoke test — just checks plumbing, not accuracy."""
    return ContestSpec(
        width=100,
        depth=16,
        n_mlps=3,
        flop_budget=10_000_000,
        ground_truth_samples=100 * 100 * 4,
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
        "schema_version": "1.0",
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
            "untracked_time_limit_s": spec.untracked_time_limit_s,
        },
    }


def _render_plain_text_report(
    report: Dict[str, Any],
    *,
    debug: bool = False,
    command: str = "run",
) -> str:
    """Render a plain-text report when Rich rendering is unavailable."""
    doc = (
        build_smoke_test_presentation(report, debug=debug)
        if command == "smoke-test"
        else build_run_presentation(report, debug=debug)
    )
    if command != "smoke-test":
        doc = replace(doc, epilogue_messages=[])
    lines = render_plain_presentation(doc).rstrip("\n").splitlines()
    if lines:
        lines[0] = "WhestBench Report (Plain Text)"

    results = report.get("results", {})

    def _extract_error_line(entry: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        raw_error = entry.get("error")
        if isinstance(raw_error, dict):
            message = str(raw_error.get("message") or "").strip()
            details = raw_error.get("details")
            if not isinstance(details, dict):
                details = {}
            if not message:
                message = "(no message)"
            return message, details
        if raw_error is None:
            return "", {}
        return str(raw_error), {}

    def _render_error_details(details: Dict[str, Any], indent: str) -> list[str]:
        if not details:
            return []
        lines: list[str] = []
        hint = details.get("hint")
        cause_hints = details.get("cause_hints")
        expected_shape = details.get("expected_shape")
        got_shape = details.get("got_shape")
        if isinstance(expected_shape, list):
            lines.append(f"{indent}Expected shape: {expected_shape}")
        if isinstance(got_shape, list):
            lines.append(f"{indent}Got shape: {got_shape}")
        if isinstance(hint, str) and hint:
            lines.append(f"{indent}Hint: {hint}")
        if isinstance(cause_hints, list) and cause_hints:
            lines.append(f"{indent}Cause hints:")
            for item in cause_hints:
                if isinstance(item, str) and item:
                    lines.append(f"{indent}  - {item}")
        return lines

    per_mlp = results.get("per_mlp") if isinstance(results, dict) else None
    if isinstance(per_mlp, list):
        failures = [e for e in per_mlp if isinstance(e, dict) and e.get("error")]
        if failures:
            lines.append("")
            lines.append(f"Estimator Errors: {len(failures)} of {len(per_mlp)} MLP(s) raised:")
            for entry in failures:
                idx = entry.get("mlp_index", "?")
                code = entry.get("error_code") or "UNKNOWN"
                message, details = _extract_error_line(entry)
                lines.append(f"  MLP {idx} [{code}]: {message}")
                lines.extend(_render_error_details(details, indent="    "))
                if debug:
                    tb_text = entry.get("traceback")
                    if tb_text:
                        for tb_line in str(tb_text).rstrip().splitlines():
                            lines.append(f"    {tb_line}")
            if not debug:
                lines.append(
                    "  (rerun with --debug to include full tracebacks; "
                    "--fail-fast to stop on first error.)"
                )

    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    run_config = report.get("run_config", {})
    n_mlps = int(run_config.get("n_mlps", 0) or 0)
    if n_mlps <= 0:
        n_mlps = 1

    breakdowns = results.get("breakdowns")
    if isinstance(breakdowns, dict):
        for breakdown_key, title in (
            ("sampling", "Sampling Budget Breakdown"),
            ("estimator", "Estimator Budget Breakdown"),
        ):
            breakdown = breakdowns.get(breakdown_key)
            if not isinstance(breakdown, dict):
                continue
            by_namespace = breakdown.get("by_namespace")
            if not isinstance(by_namespace, dict):
                by_namespace = {}

            total_flops = _as_float(breakdown.get("flops_used", 0.0))
            if total_flops <= 0.0:
                total_flops = sum(
                    _as_float(bucket.get("flops_used", 0.0))
                    for bucket in by_namespace.values()
                    if isinstance(bucket, dict)
                )

            lines.append("")
            lines.append(f"{title}:")

            if breakdown_key == "estimator":
                # FLOP budget gauge (ASCII mirror of the Rich gauge)
                gauge = _compute_gauge_state(report)
                if not gauge.has_budget:
                    lines.append("  Estimator FLOPs  -- of 0 FLOPs")
                else:
                    bar_rich = _gauge_bar_fragment(gauge.mean_utilization)
                    bar_ascii = bar_rich.replace("█", "#").replace("░", "-")
                    overflow = ">" if gauge.state_name == "catastrophic" else ""
                    pct_int = int(gauge.mean_utilization * 100)
                    budget_label = _fmt_flops(gauge.flop_budget)
                    suffix = (
                        f" . worst MLP {gauge.worst_mlp_pct}% !"
                        if gauge.worst_mlp_pct is not None
                        else ""
                    )
                    lines.append(
                        f"  Estimator FLOPs  {bar_ascii}{overflow} {pct_int}% of "
                        f"{budget_label}{suffix}"
                    )

                # Over-Budget MLPs (ASCII mirror of the Rich panel)
                selection = _select_top_over_budget(report)
                if selection.busted_count > 0:
                    lines.append("  Over-Budget MLPs")
                    for row in selection.rows:
                        pct_label = (
                            f"{row.pct_of_budget}%" if row.pct_of_budget is not None else "--%"
                        )
                        lines.append(
                            f"    MLP #{row.mlp_index:<4}  "
                            f"{_fmt_flops(row.flops_used):>9} FLOPs  "
                            f"{pct_label:>4} of budget  zeroed"
                        )
                    if selection.is_truncated:
                        remainder = selection.busted_count - len(selection.rows)
                        lines.append(f"    ... and {remainder} more over budget")
                        lines.append("    run with --json for the full list")
                    if selection.is_all_busted:
                        lines.append(
                            f"    All {selection.n_mlps} MLPs exceeded the per-MLP FLOP cap "
                            f"— predictions entirely zeroed"
                        )
                    else:
                        lines.append(
                            f"    {selection.busted_count} of {selection.n_mlps} MLPs "
                            f"exceeded the per-MLP FLOP cap"
                        )

            lines.extend(
                [
                    f"  Total FLOPs: {_fmt_flops(total_flops)}",
                    f"  Tracked Time: {_as_float(breakdown.get('tracked_time_s', 0.0)):.6f}s",
                    f"  Untracked Time: {_as_float(breakdown.get('untracked_time_s', 0.0)):.6f}s",
                ]
            )

            for namespace, bucket in sorted(
                (
                    (namespace, bucket)
                    for namespace, bucket in by_namespace.items()
                    if isinstance(bucket, dict)
                ),
                key=lambda item: _as_float(item[1].get("flops_used", 0.0)),
                reverse=True,
            ):
                namespace_label = (
                    "(unlabeled)" if namespace in {None, "", "null", "None"} else str(namespace)
                )
                flops_used = _as_float(bucket.get("flops_used", 0.0))
                percent = (flops_used / total_flops * 100.0) if total_flops > 0 else 0.0
                mean_flops = flops_used / n_mlps if n_mlps > 0 else 0.0
                tracked_time_s = _as_float(bucket.get("tracked_time_s", 0.0))
                lines.append(
                    "  "
                    f"{namespace_label}: {_fmt_flops(flops_used)} FLOPs ({percent:.1f}%), "
                    f"mean {_fmt_flops(mean_flops)}/MLP, tracked {tracked_time_s:.6f}s"
                )
    return "\n".join(lines) + "\n"


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
        "schema_version": "1.0",
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
            "untracked_time_limit_s": contest_spec.untracked_time_limit_s,
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


class _LiveTopPaneSession:
    def __init__(
        self,
        pre_report: Dict[str, Any],
        total: int,
        n_mlps: int,
        gen_label: str = "Generating MLPs",
    ) -> None:
        self._pre_report = pre_report
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        self._gen_task_id = self._progress.add_task(gen_label, total=n_mlps)
        # Scoring task is added hidden and unstarted so its elapsed timer
        # doesn't tick during the generating phase. It is revealed and
        # started on the first "scoring" progress event.
        self._scoring_task_id = self._progress.add_task(
            "Scoring", total=total, start=False, visible=False
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
        completed = int(event.get("completed", 0))
        if completed <= self._last_completed.get(phase, -1):
            return
        self._last_completed[phase] = completed
        total_value = event.get("total")
        force_refresh = False
        if phase == "generating":
            total_update = int(total_value) if isinstance(total_value, int) else None
            self._progress.update(
                self._gen_task_id,
                completed=completed,
                total=total_update,
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
                total=n_mlps,
                desc=gen_label,
                unit="mlp",
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
            if phase == "generating":
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

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    gen_task_id = progress.add_task(gen_label, total=n_mlps)
    # Scoring task starts hidden and unstarted — revealed on first scoring event.
    scoring_task_id = progress.add_task("Scoring", total=total, start=False, visible=False)
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
        completed = int(event.get("completed", 0))
        if completed <= state.get(phase, -1):
            return
        state[phase] = completed
        total_value = event.get("total")
        force_refresh = False
        if phase == "generating":
            total_update = int(total_value) if isinstance(total_value, int) else None
            progress.update(
                gen_task_id,
                completed=completed,
                total=total_update,
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
        template = Path(__file__).resolve().parent / "templates" / "estimator.py.tmpl"
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
) -> Dict[str, Any]:
    estimator, metadata = load_estimator_from_path(estimator_path, class_name=class_name)
    context = SetupContext(width=4, depth=2, flop_budget=100, api_version="1.0")
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
) -> Dict[str, Any]:
    result = _run_validate_checks(estimator_path, class_name=class_name)
    return {
        "ok": result["ok"],
        "class_name": result["class_name"],
        "module_name": result["module_name"],
        "output_shape": result["output_shape"],
    }


def _build_participant_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Participant-first WhestBench CLI. Starter examples live in examples/estimators/."
        )
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
    smoke_test_parser.add_argument(
        "--no-rich",
        action="store_true",
        help=(
            "Disable Rich live display and progress bars; use plain-text output. "
            "Use when attaching a debugger (pdb/ipdb) or running in a non-TTY environment."
        ),
    )
    smoke_test_parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        metavar="N",
        help="Limit BLAS to at most N CPU threads.",
    )

    init_parser = subparsers.add_parser("init", help="Create starter estimator files.")
    init_parser.add_argument("path", nargs="?", default=".")
    init_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Return results as a JSON string."
    )
    init_parser.add_argument("--debug", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate estimator contract.")
    validate_parser.add_argument(
        "--estimator",
        required=True,
        help="Path to estimator.py (see examples/estimators/ for starter files).",
    )
    validate_parser.add_argument("--class", dest="class_name")
    validate_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Return results as a JSON string."
    )
    validate_parser.add_argument("--debug", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run local evaluation for an estimator.")
    run_parser.add_argument(
        "--estimator",
        required=True,
        help="Path to estimator.py (see examples/estimators/ for starter files).",
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
    run_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Return results as a JSON string."
    )
    run_parser.add_argument(
        "--dataset", default=None, help="Path to pre-created dataset .npz file."
    )
    run_parser.add_argument(
        "--flop-budget",
        type=int,
        default=None,
        metavar="N",
        help="FLOP budget for estimator predict calls (default: 100_000_000).",
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
        default=None,
        metavar="SECONDS",
        help="Wall-clock time limit per predict call (default: unlimited).",
    )
    run_parser.add_argument(
        "--untracked-time-limit",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Time limit for non-whest operations per predict call (default: unlimited).",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic MLP generation and ground-truth sampling (no-dataset runs only).",
    )
    run_parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        metavar="N",
        help="Limit BLAS to at most N CPU threads.",
    )
    run_parser.add_argument(
        "--no-rich",
        action="store_true",
        help=(
            "Disable Rich live display and progress bars; use plain-text output. "
            "Use when attaching a debugger (pdb/ipdb) or running in a non-TTY environment."
        ),
    )

    create_ds_parser = subparsers.add_parser(
        "create-dataset", help="Pre-create evaluation dataset."
    )
    create_ds_parser.add_argument("--n-mlps", type=int, default=10)
    create_ds_parser.add_argument("--n-samples", type=int, default=10000)
    create_ds_parser.add_argument("--width", type=int, default=None)
    create_ds_parser.add_argument("--depth", type=int, default=None)
    create_ds_parser.add_argument("--flop-budget", type=int, default=None)
    create_ds_parser.add_argument("--seed", type=int, default=None)
    create_ds_parser.add_argument("-o", "--output", default="eval_dataset.npz")
    create_ds_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Return results as a JSON string."
    )
    create_ds_parser.add_argument("--debug", action="store_true")
    create_ds_parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        metavar="N",
        help="Limit BLAS to at most N CPU threads.",
    )

    package_parser = subparsers.add_parser("package", help="Package submission artifact.")
    package_parser.add_argument("--estimator", required=True)
    package_parser.add_argument("--class", dest="class_name")
    package_parser.add_argument("--requirements")
    package_parser.add_argument("--submission-metadata")
    package_parser.add_argument("--approach")
    package_parser.add_argument("--output")
    package_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Return results as a JSON string."
    )
    package_parser.add_argument("--debug", action="store_true")

    visualizer_parser = subparsers.add_parser(
        "visualizer",
        help="Launch the interactive WhestBench Explorer in a browser.",
    )
    visualizer_parser.add_argument(
        "--host", default="localhost", help="Bind address (default: localhost)."
    )
    visualizer_parser.add_argument(
        "--port", type=int, default=5173, help="Port number (default: 5173)."
    )
    visualizer_parser.add_argument(
        "--no-open", action="store_true", help="Don't auto-open browser."
    )
    visualizer_parser.add_argument("--debug", action="store_true")

    profile_parser = subparsers.add_parser(
        "profile-simulation",
        help="Benchmark whest simulation performance.",
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

    return parser


def _normalize_runner_name(raw_runner: str) -> str:
    """Normalize legacy/alias runner names to canonical internal names."""
    if raw_runner == "server":
        return "subprocess"
    if raw_runner == "inprocess":
        return "local"
    return raw_runner


class _RunnerEstimator(BaseEstimator):
    """Adapter that wraps a started runner as a BaseEstimator for scoring."""

    def __init__(self, runner: "Any") -> None:
        self._runner = runner

    def predict(self, mlp: "Any", budget: int) -> we.ndarray:
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
            "tracked_time_s": getattr(stats, "tracked_time_s", None),
            "untracked_time_s": getattr(stats, "untracked_time_s", None),
            "budget_breakdown": getattr(stats, "budget_breakdown", None),
        }


def _run_estimator_with_runner(
    runner: "Any",
    *,
    entrypoint: EstimatorEntrypoint,
    contest_spec: ContestSpec,
    n_mlps: int,
    profile: bool,
    detail: str,
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
            on_mlp_done=lambda i: progress(
                {"phase": "generating", "completed": i, "total": spec.n_mlps}
            ),
        )
    else:
        data = make_contest(spec)

    context = SetupContext(
        width=spec.width,
        depth=spec.depth,
        flop_budget=spec.flop_budget,
        api_version="1.0",
    )
    limits = ResourceLimits(
        setup_timeout_s=spec.setup_timeout_s,
        predict_timeout_s=spec.predict_timeout_s,
        memory_limit_mb=spec.memory_limit_mb,
        flop_budget=spec.flop_budget,
        wall_time_limit_s=spec.wall_time_limit_s,
        untracked_time_limit_s=spec.untracked_time_limit_s,
    )

    t0 = _time.time()
    runner.start(entrypoint, context, limits)

    try:
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
    return {
        "schema_version": "1.0",
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
            "untracked_time_limit_s": spec.untracked_time_limit_s,
        },
    }


def _main_participant(argv: "list[str]") -> int:
    args = _build_participant_parser().parse_args(argv)
    command = str(args.command)
    json_output = bool(getattr(args, "json_output", False))

    # Apply thread limit early, before any backend module is imported.
    max_threads = getattr(args, "max_threads", None)
    if max_threads is not None:
        from .concurrency import apply_thread_limit

        apply_thread_limit(max_threads)
    debug = bool(getattr(args, "debug", False))

    # Promote debugger detection to --no-rich so Live displays don't mask pdb
    # prompts. Only affects subcommands that expose the flag (run, smoke-test);
    # other subcommands silently ignore it via getattr defaults.
    if (
        hasattr(args, "no_rich")
        and not bool(getattr(args, "no_rich", False))
        and _debugger_active()
    ):
        args.no_rich = True
        print(
            "Debugger detected (sys.gettrace / PYTHONBREAKPOINT / WHESTBENCH_NO_RICH); "
            "disabling Rich live display.",
            file=sys.stderr,
        )
    no_rich = bool(getattr(args, "no_rich", False))

    try:
        if command == "smoke-test":
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
                    presentation_doc=replace(smoke_doc, epilogue_messages=[]),
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
            return 0

        if command == "init":
            created = _write_init_template(Path(args.path).resolve())
            payload = {"ok": True, "created": created}
            if json_output:
                print(json.dumps(payload, indent=2))
            else:
                print(render_plain_presentation(build_init_presentation(payload)), end="")
            return 0

        if command == "validate":
            if json_output:
                payload = validate_submission_entrypoint(args.estimator, class_name=args.class_name)
                print(json.dumps(payload, indent=2))
            else:
                result = _run_validate_checks(args.estimator, class_name=args.class_name)
                print(render_rich_presentation(build_validate_presentation(result)), end="")
            return 0

        if command == "create-dataset":
            from .dataset import create_dataset as _create_dataset

            contest = _default_contest_spec()
            ds_width = args.width or contest.width
            ds_depth = args.depth or contest.depth
            ds_flop_budget = args.flop_budget or contest.flop_budget
            n_mlps_ds = int(args.n_mlps)

            if not json_output:
                try:
                    from rich.progress import (
                        BarColumn,
                        MofNCompleteColumn,
                        Progress,
                        SpinnerColumn,
                        TextColumn,
                        TimeElapsedColumn,
                    )

                    progress_bar = Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(bar_width=None),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                    )
                    gen_task = progress_bar.add_task("Generating MLPs", total=n_mlps_ds)
                    sample_task = progress_bar.add_task("Sampling ground truth", total=n_mlps_ds)

                    def _on_ds_progress(event: Dict[str, Any]) -> None:
                        phase = str(event.get("phase", ""))
                        completed = int(event.get("completed", 0))
                        if phase == "generating":
                            progress_bar.update(gen_task, completed=completed)
                        elif phase == "sampling":
                            progress_bar.update(sample_task, completed=completed)

                    with progress_bar:
                        out = _create_dataset(
                            n_mlps=n_mlps_ds,
                            n_samples=int(args.n_samples),
                            width=ds_width,
                            depth=ds_depth,
                            flop_budget=ds_flop_budget,
                            seed=getattr(args, "seed", None),
                            output_path=Path(args.output),
                            progress=_on_ds_progress,
                        )
                except ImportError:
                    out = _create_dataset(
                        n_mlps=n_mlps_ds,
                        n_samples=int(args.n_samples),
                        width=ds_width,
                        depth=ds_depth,
                        flop_budget=ds_flop_budget,
                        seed=getattr(args, "seed", None),
                        output_path=Path(args.output),
                    )
            else:
                out = _create_dataset(
                    n_mlps=n_mlps_ds,
                    n_samples=int(args.n_samples),
                    width=ds_width,
                    depth=ds_depth,
                    flop_budget=ds_flop_budget,
                    seed=getattr(args, "seed", None),
                    output_path=Path(args.output),
                )
            payload = {"ok": True, "path": str(out)}
            if json_output:
                print(json.dumps(payload, indent=2))
            else:
                print(render_plain_presentation(build_create_dataset_presentation(payload)), end="")
            return 0

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
            bundle = None
            ds_meta: Dict[str, Any] = {}
            if getattr(args, "dataset", None) is not None and run_seed is not None:
                raise ValueError("--seed is only valid when --dataset is not provided.")

            if dataset_path is not None:
                from .dataset import dataset_file_hash, load_dataset
                from .scoring import make_contest_from_bundle

                bundle = load_dataset(dataset_path)
                ds_meta = bundle.metadata

                if user_n_mlps is None:
                    n_mlps = bundle.n_mlps
                elif user_n_mlps > bundle.n_mlps:
                    print(
                        f"Warning: --n-mlps={user_n_mlps} exceeds dataset size "
                        f"({bundle.n_mlps}); using {bundle.n_mlps}.",
                        file=sys.stderr,
                    )
                    n_mlps = bundle.n_mlps
                else:
                    n_mlps = user_n_mlps

                contest_spec = ContestSpec(
                    width=ds_meta["width"],
                    depth=ds_meta["depth"],
                    n_mlps=n_mlps,
                    flop_budget=ds_meta.get("flop_budget", flop_budget),
                    ground_truth_samples=gt_samples,
                    seed=None,
                    wall_time_limit_s=getattr(args, "wall_time_limit", None),
                    untracked_time_limit_s=getattr(args, "untracked_time_limit", None),
                )
                contest_data = make_contest_from_bundle(contest_spec, bundle, n_mlps)
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
                    untracked_time_limit_s=getattr(args, "untracked_time_limit", None),
                )

            score_kwargs: Dict[str, Any] = {
                "entrypoint": entrypoint,
                "contest_spec": contest_spec,
                "n_mlps": n_mlps,
                "profile": bool(args.profile),
                "detail": str(args.detail),
                "contest_data": contest_data,
                "fail_fast": bool(getattr(args, "fail_fast", False)),
            }

            if json_output:
                report = _run_estimator_with_runner(runner, **score_kwargs)
                report["mode"] = "agent"
                if dataset_path is not None:
                    report.setdefault("run_config", {})["dataset"] = {
                        "path": str(Path(dataset_path).resolve()),
                        "sha256": dataset_file_hash(dataset_path),
                        "seed": ds_meta.get("seed"),
                        "n_mlps": ds_meta.get("n_mlps"),
                    }
                output = render_agent_report(report)
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
                total_units = n_mlps
                _dataset_tip = (
                    "\n[bold bright_yellow]Tip:[/] Ground truth is recomputed on every run. "
                    "Consider creating and reusing a dataset:\n"
                    "   [cyan]whest create-dataset[/] [green]--n-mlps[/] [yellow]10[/] [green]--n-samples[/] [yellow]10000[/] [green]-o[/] [yellow]my_dataset.npz[/]\n"
                    "   [cyan]whest run[/] [green]--estimator[/] [yellow]...[/] [green]--dataset[/] [yellow]my_dataset.npz[/]\n"
                )
                _tip_console = Console(highlight=False)
                gen_label = "Loading dataset" if dataset_path is not None else "Generating MLPs"
                if no_rich:
                    # Plain-text path: skip every Rich Live/progress display so
                    # breakpoints in predict() reach a clean terminal. Emit
                    # single-line phase updates to stderr so logs still show
                    # progress without terminal control sequences.
                    print(f"[run] estimator_class={metadata.class_name}", file=sys.stderr)
                    print(f"[run] estimator_path={args.estimator}", file=sys.stderr)
                    print(f"[run] n_mlps={n_mlps} gen_label={gen_label}", file=sys.stderr)
                    _plain_state = {"gen": -1, "scoring": -1}

                    def _plain_run_progress(event: Dict[str, Any]) -> None:
                        phase = str(event.get("phase", "scoring"))
                        completed = int(event.get("completed", 0))
                        total_value = event.get("total")
                        total = int(total_value) if isinstance(total_value, int) else n_mlps
                        # Only print when the completed count changes, to
                        # avoid flooding stderr with duplicates.
                        key = "gen" if phase == "generating" else "scoring"
                        if completed == _plain_state.get(key):
                            return
                        _plain_state[key] = completed
                        print(f"[run] {phase}: {completed}/{total}", file=sys.stderr)

                    score_kwargs["progress"] = _plain_run_progress
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
                    report.setdefault("run_config", {})["dataset"] = {
                        "path": str(Path(dataset_path).resolve()),
                        "sha256": dataset_file_hash(dataset_path),
                        "seed": ds_meta.get("seed"),
                        "n_mlps": ds_meta.get("n_mlps"),
                    }
                if no_rich:
                    output = _render_plain_text_report(report, debug=debug)
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
                        output = _render_plain_text_report(report, debug=debug)
            print(output, end="" if output.endswith("\n") else "\n")
            if not json_output and not no_rich:
                _tip_console.print(
                    "[bold bright_yellow]Tip:[/] Use [green]--json[/] for JSON output when calling from automated agents or UIs."
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
            artifact_path = package_submission(
                args.estimator,
                class_name=args.class_name,
                requirements_path=args.requirements,
                submission_yaml_path=args.submission_metadata,
                approach_md_path=args.approach,
                output_path=args.output,
            )
            payload = {"ok": True, "artifact_path": str(artifact_path)}
            if json_output:
                print(json.dumps(payload, indent=2))
            else:
                print(render_plain_presentation(build_package_presentation(payload)), end="")
            return 0

        if command == "visualizer":
            from . import visualizer as _viz_mod

            return _viz_mod.run_visualizer(
                host=str(args.host),
                port=int(args.port),
                no_open=bool(args.no_open),
                debug=bool(args.debug),
            )

        if command == "profile-simulation":
            from .profiler import run_profile

            terminal_output, _ = run_profile(
                preset_name=str(args.preset),
                output_path=args.output,
                show_progress=not json_output,
                max_threads=args.max_threads,
                verbose=bool(args.verbose),
                log_progress=bool(args.log_progress),
            )
            print(terminal_output)
            return 0

        raise ValueError(f"Unsupported command: {command}")
    except Exception as exc:  # pragma: no cover - exercised by CLI tests
        stage = exc.stage if isinstance(exc, RunnerError) else command
        payload = _error_payload(exc, include_traceback=debug, stage=stage)
        _print_error(
            payload,
            json_output=json_output,
            debug=debug,
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
    show_inprocess_hint: bool = False,
) -> None:
    if json_output:
        print(json.dumps(payload, indent=2))
        return
    doc = build_error_presentation(
        payload,
        debug=debug,
        show_inprocess_hint=show_inprocess_hint,
    )
    print(render_plain_presentation(doc), end="")


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
