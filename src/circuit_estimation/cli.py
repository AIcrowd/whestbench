"""CLI and convenience entrypoints for local scoring runs."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, cast, overload

import numpy as np
from numpy.typing import NDArray
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm as classic_tqdm
from tqdm.std import TqdmExperimentalWarning

try:
    from tqdm.rich import tqdm as rich_tqdm
except Exception:  # pragma: no cover - optional at runtime
    rich_tqdm = None

from .domain import Circuit, Layer
from .estimators import CombinedEstimator
from .hardware import collect_hardware_fingerprint
from .loader import load_estimator_from_path, resolve_estimator_class_metadata
from .packaging import package_submission
from .reporting import (
    build_human_context_renderable,
    render_agent_report,
    render_human_context_panels,
    render_human_header,
    render_human_report,
    render_human_results,
    render_smoke_test_next_steps,
)
from .runner import (
    EstimatorEntrypoint,
    InProcessRunner,
    ResourceLimits,
    RunnerError,
    SubprocessRunner,
)
from .scoring import ContestParams, score_estimator_report
from .sdk import SetupContext
from .streaming import validate_depth_row

_DEFAULT_ESTIMATOR = CombinedEstimator()
ProgressCallback = Callable[[dict[str, int | str]], None]


def _default_contest_params() -> ContestParams:
    return ContestParams(
        width=100,
        max_depth=30,
        budgets=[10, 100, 1000, 10000],
        time_tolerance=0.1,
    )


def _default_resource_limits() -> ResourceLimits:
    return ResourceLimits(
        setup_timeout_s=5.0,
        predict_timeout_s=30.0,
        memory_limit_mb=4096,
        cpu_time_limit_s=None,
    )


@overload
def run_default_score(profile: Literal[False] = False) -> float: ...


@overload
def run_default_score(profile: Literal[True]) -> tuple[float, list[dict[str, Any]]]: ...


def run_default_score(profile: bool = False) -> float | tuple[float, list[dict[str, Any]]]:
    """Run the built-in smoke-test scenario and return score-only output.

    When ``profile`` is true, this returns ``(score, profile_calls)``.
    """
    report = run_default_report(profile=profile, detail="raw")
    score = float(report["results"]["final_score"])
    if profile:
        return score, list(report.get("profile_calls", []))
    return score


def run_default_report(*, profile: bool = False, detail: str = "raw") -> dict[str, Any]:
    """Run the built-in smoke-test scenario and return report payload."""
    return score_estimator_report(
        _DEFAULT_ESTIMATOR.predict,
        n_circuits=10,
        n_samples=10000,
        contest_params=_default_contest_params(),
        profile=profile,
        detail=detail,
    )


def _render_plain_text_report(report: dict[str, Any]) -> str:
    """Render a minimal plain-text summary when Rich rendering is unavailable."""
    results = report.get("results", {})
    run_config = report.get("run_config", {})
    run_meta = report.get("run_meta", {})
    best_budget_score, worst_budget_score = _budget_score_bounds(results.get("by_budget_raw"))
    lines = [
        "Circuit Estimation Report (Plain Text)",
        f"Final Score: {results.get('final_score', 'n/a')}",
        f"Best Budget Score: {best_budget_score}",
        f"Worst Budget Score: {worst_budget_score}",
        f"Duration(s): {run_meta.get('run_duration_s', 'n/a')}",
        f"Circuits: {run_config.get('n_circuits', 'n/a')}",
        f"Samples/Circuit: {run_config.get('n_samples', 'n/a')}",
        f"Width: {run_config.get('width', 'n/a')}",
        f"Max Depth: {run_config.get('max_depth', 'n/a')}",
        f"Budgets: {run_config.get('budgets', 'n/a')}",
    ]
    return "\n".join(lines) + "\n"


def _budget_score_bounds(by_budget_raw: object) -> tuple[float | str, float | str]:
    scores: list[float] = []
    if isinstance(by_budget_raw, list):
        for item in by_budget_raw:
            if not isinstance(item, dict):
                continue
            score = item.get("adjusted_mse", item.get("score"))
            if isinstance(score, (int, float)):
                scores.append(float(score))
    if not scores:
        return "n/a", "n/a"
    return min(scores), max(scores)


def _host_metadata() -> dict[str, Any]:
    return collect_hardware_fingerprint()


def _fmt_ram(value: int | None) -> str:
    """Format byte count as human-readable string for warning messages."""
    if value is None:
        return "unknown RAM"
    gb = value / (1024 ** 3)
    return f"{gb:.0f}GB"


def _pre_run_report(
    *,
    n_circuits: int,
    n_samples: int,
    contest_params: ContestParams,
    profile: bool,
    detail: str,
    estimator_class: str,
    estimator_path: str,
) -> dict[str, Any]:
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
            "n_circuits": int(n_circuits),
            "n_samples": int(n_samples),
            "width": int(contest_params.width),
            "max_depth": int(contest_params.max_depth),
            "layer_count": int(contest_params.max_depth),
            "budgets": [int(b) for b in contest_params.budgets],
            "time_tolerance": float(contest_params.time_tolerance),
            "profile_enabled": bool(profile),
            "estimator_class": estimator_class,
            "estimator_path": estimator_path,
        },
        "results": {"by_budget_raw": []},
    }


def _print_human_startup(
    pre_report: dict[str, Any],
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
    print("Use --json for JSON output when calling from automated agents or UIs.")
    print("Use --show-diagnostic-plots to include diagnostic plot panes.")
    print("Runtime scoring uses budget-by-depth checks at each streamed predict() row.")


class _LiveTopPaneSession:
    def __init__(self, pre_report: dict[str, Any], total: int) -> None:
        self._pre_report = pre_report
        self._progress = Progress(
            TextColumn("[bold bright_yellow]{task.description}[/]"),
            BarColumn(bar_width=None, complete_style="bright_green", finished_style="bright_green"),
            TaskProgressColumn(show_speed=True),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self._sampling_task_id = self._progress.add_task("Sampling (Ground Truth)", total=None)
        self._scoring_task_id = self._progress.add_task("Scoring", total=total)
        self._live = Live(
            self._renderable(),
            console=None,
            refresh_per_second=8,
            transient=False,
        )

    def _renderable(self) -> Group:
        context = build_human_context_renderable(self._pre_report)
        progress_panel = Panel(
            self._progress,
            title="[bold bright_yellow]Progress[/]",
            border_style="bright_yellow",
        )
        return Group(context, progress_panel)

    def on_progress(self, event: dict[str, int | str]) -> None:
        phase = str(event.get("phase", "scoring"))
        completed = int(event.get("completed", 0))
        total_value = event.get("total")
        if phase == "sampling":
            total_update = int(total_value) if isinstance(total_value, int) else None
            self._progress.update(self._sampling_task_id, completed=completed, total=total_update)
            return
        self._progress.update(self._scoring_task_id, completed=completed)

    def update_run_meta(self, run_meta: dict[str, Any]) -> None:
        current_meta = self._pre_report.get("run_meta")
        if isinstance(current_meta, dict):
            current_meta.update(run_meta)
        self._live.update(self._renderable())

    def start(self) -> None:
        self._live.start()

    def stop(self) -> None:
        self._live.stop()
        print()


@contextmanager
def _live_top_pane_session(pre_report: dict[str, Any], total: int) -> Iterator[_LiveTopPaneSession]:
    session = _LiveTopPaneSession(pre_report, total)
    session.start()
    try:
        yield session
    finally:
        session.stop()


@contextmanager
def _progress_callback(total: int) -> Iterator[ProgressCallback]:
    if rich_tqdm is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", TqdmExperimentalWarning)
            sampling_bar = classic_tqdm(
                total=None,
                desc="Sampling (Ground Truth)",
                unit="circuit",
                file=sys.stdout,
                position=0,
            )
            scoring_bar = classic_tqdm(
                total=total,
                desc="Scoring",
                unit="eval",
                file=sys.stdout,
                position=1,
            )

        state = {"sampling_completed": 0, "scoring_completed": 0}

        def _on_progress(event: dict[str, int | str]) -> None:
            phase = str(event.get("phase", "scoring"))
            completed = int(event.get("completed", 0))
            if phase == "sampling":
                total_value = event.get("total")
                if isinstance(total_value, int) and sampling_bar.total != total_value:
                    sampling_bar.total = total_value
                    sampling_bar.refresh()
                delta = completed - state["sampling_completed"]
                if delta > 0:
                    sampling_bar.update(delta)
                    state["sampling_completed"] = completed
                return
            delta = completed - state["scoring_completed"]
            if delta > 0:
                scoring_bar.update(delta)
                state["scoring_completed"] = completed

        try:
            yield _on_progress
        finally:
            sampling_bar.close()
            scoring_bar.close()
            print()
        return

    progress = Progress(
        TextColumn("[bold bright_yellow]{task.description}[/]"),
        BarColumn(bar_width=None, complete_style="bright_green", finished_style="bright_green"),
        TaskProgressColumn(show_speed=True),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    sampling_task_id = progress.add_task("Sampling (Ground Truth)", total=None)
    scoring_task_id = progress.add_task("Scoring", total=total)
    live = Live(
        Panel(progress, title="[bold bright_yellow]Progress[/]", border_style="bright_yellow"),
        console=None,
        refresh_per_second=8,
        transient=False,
    )

    def _on_progress(event: dict[str, int | str]) -> None:
        phase = str(event.get("phase", "scoring"))
        completed = int(event.get("completed", 0))
        total_value = event.get("total")
        if phase == "sampling":
            total_update = int(total_value) if isinstance(total_value, int) else None
            progress.update(sampling_task_id, completed=completed, total=total_update)
            return
        progress.update(scoring_task_id, completed=completed)

    try:
        live.start()
        yield _on_progress
    finally:
        live.stop()
        print()


def _write_init_template(target_dir: Path) -> list[str]:
    created: list[str] = []
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


def _consume_prediction_stream(
    predictions: object,
    *,
    width: int,
    depth: int,
) -> NDArray[np.float32]:
    try:
        output_iter = iter(cast(Any, predictions))
    except TypeError as exc:
        raise ValueError("Estimator must return an iterator of depth-row outputs.") from exc

    rows: list[NDArray[np.float32]] = []
    for depth_index in range(depth):
        try:
            raw_row = next(output_iter)
        except StopIteration as exc:
            raise ValueError("Estimator must emit exactly max_depth rows.") from exc
        rows.append(validate_depth_row(raw_row, width=width, depth_index=depth_index))

    try:
        _extra = next(output_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("Estimator emitted more than max_depth rows.")

    return np.stack(rows, axis=0).astype(np.float32)


def validate_submission_entrypoint(
    estimator_path: str | Path,
    *,
    class_name: str | None = None,
) -> dict[str, Any]:
    estimator, metadata = load_estimator_from_path(estimator_path, class_name=class_name)
    context = SetupContext(
        width=4,
        max_depth=1,
        budgets=(10,),
        time_tolerance=0.1,
        api_version="1.0",
    )
    circuit = Circuit(n=4, d=1, gates=[Layer.identity(4)])

    try:
        estimator.setup(context)
        predictions = estimator.predict(circuit, 10)
        tensor = _consume_prediction_stream(predictions, width=circuit.n, depth=circuit.d)
    finally:
        estimator.teardown()

    return {
        "ok": True,
        "class_name": metadata.class_name,
        "module_name": metadata.module_name,
        "output_shape": list(tensor.shape),
    }


def _build_participant_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Participant-first circuit-estimation CLI. Starter examples live in "
            "examples/estimators/."
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
    run_parser.add_argument("--runner", choices=("inprocess", "subprocess"), default="subprocess")
    run_parser.add_argument("--n-circuits", type=int, default=10)
    run_parser.add_argument("--n-samples", type=int, default=10000)
    run_parser.add_argument("--detail", choices=("raw", "full"), default="raw")
    run_parser.add_argument("--profile", action="store_true")
    run_parser.add_argument("--show-diagnostic-plots", action="store_true")
    run_parser.add_argument("--json", dest="json_output", action="store_true", help="Return results as a JSON string.")
    run_parser.add_argument("--dataset", default=None, help="Path to pre-created dataset .npz file.")
    run_parser.add_argument("--strict-baselines", action="store_true", help="Refuse to run if dataset hardware differs.")
    run_parser.add_argument("--debug", action="store_true")

    create_ds_parser = subparsers.add_parser(
        "create-dataset", help="Pre-create evaluation dataset."
    )
    create_ds_parser.add_argument("--n-circuits", type=int, default=10)
    create_ds_parser.add_argument("--n-samples", type=int, default=10000)
    create_ds_parser.add_argument("--width", type=int, default=None)
    create_ds_parser.add_argument("--max-depth", type=int, default=None)
    create_ds_parser.add_argument("--budgets", type=str, default=None)
    create_ds_parser.add_argument("--seed", type=int, default=None)
    create_ds_parser.add_argument("-o", "--output", default="eval_dataset.npz")
    create_ds_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Return results as a JSON string."
    )
    create_ds_parser.add_argument("--debug", action="store_true")

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

    return parser


def _main_participant(argv: list[str]) -> int:
    args = _build_participant_parser().parse_args(argv)
    command = str(args.command)
    json_output = bool(getattr(args, "json_output", False))
    debug = bool(getattr(args, "debug", False))

    try:
        if command == "smoke-test":
            report = run_default_report(profile=bool(args.profile), detail=str(args.detail))
            report["mode"] = "human"
            try:
                output = render_human_report(
                    report, show_diagnostic_plots=bool(args.show_diagnostic_plots)
                )
            except Exception as exc:
                print(
                    f"Rich dashboard unavailable ({exc}); falling back to plain-text report.",
                    file=sys.stderr,
                )
                output = _render_plain_text_report(report)
            print(output, end="" if output.endswith("\n") else "\n")
            next_steps = render_smoke_test_next_steps()
            print(next_steps, end="" if next_steps.endswith("\n") else "\n")
            return 0

        if command == "init":
            created = _write_init_template(Path(args.path).resolve())
            payload = {"ok": True, "created": created}
            if json_output:
                print(json.dumps(payload, indent=2))
            elif created:
                print("Initialized starter files:")
                for file in created:
                    print(f"- {file}")
            else:
                print("Starter files already exist; nothing created.")
            return 0

        if command == "validate":
            payload = validate_submission_entrypoint(args.estimator, class_name=args.class_name)
            if json_output:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"Validation passed: class={payload['class_name']} "
                    f"shape={tuple(payload['output_shape'])}"
                )
            return 0

        if command == "create-dataset":
            from .dataset import create_dataset as _create_dataset

            contest = _default_contest_params()
            ds_width = args.width or contest.width
            ds_max_depth = args.max_depth or contest.max_depth
            ds_budgets = (
                [int(b) for b in args.budgets.split(",")]
                if args.budgets
                else list(contest.budgets)
            )
            out = _create_dataset(
                n_circuits=int(args.n_circuits),
                n_samples=int(args.n_samples),
                width=ds_width,
                max_depth=ds_max_depth,
                budgets=ds_budgets,
                seed=getattr(args, "seed", None),
                output_path=Path(args.output),
            )
            payload = {"ok": True, "path": str(out)}
            if json_output:
                print(json.dumps(payload, indent=2))
            else:
                print(f"Dataset created: {out}")
            return 0

        if command == "run":
            runner = InProcessRunner() if args.runner == "inprocess" else SubprocessRunner()
            contest_params = _default_contest_params()
            n_circuits = int(args.n_circuits)
            n_samples = int(args.n_samples)
            entrypoint = EstimatorEntrypoint(
                file_path=Path(args.estimator).resolve(),
                class_name=args.class_name,
            )

            # --- Dataset loading ---
            preloaded_circuits: list[Circuit] | None = None
            preloaded_means: Any = None
            preloaded_baselines: dict[int, Any] | None = None
            baselines_recomputed = False
            dataset_path: str | None = getattr(args, "dataset", None)

            if dataset_path is not None:
                from .dataset import dataset_file_hash, load_dataset
                from .hardware import hardware_matches

                bundle = load_dataset(dataset_path)
                ds_meta = bundle.metadata
                # Override contest params from dataset
                contest_params = ContestParams(
                    width=ds_meta["width"],
                    max_depth=ds_meta["max_depth"],
                    budgets=ds_meta["budgets"],
                    time_tolerance=ds_meta.get("time_tolerance", contest_params.time_tolerance),
                )
                n_circuits = bundle.n_circuits
                preloaded_circuits = bundle.circuits
                preloaded_means = bundle.ground_truth_means

                # Hardware staleness check
                current_hw = collect_hardware_fingerprint()
                stored_hw = ds_meta.get("hardware", {})
                if not hardware_matches(stored_hw, current_hw):
                    if getattr(args, "strict_baselines", False):
                        msg = (
                            "Hardware mismatch detected and --strict-baselines is set.\n"
                            f"  Dataset host: {stored_hw.get('hostname', 'unknown')} "
                            f"({stored_hw.get('cpu_brand', '?')}, "
                            f"{stored_hw.get('cpu_count_logical', '?')} cores)\n"
                            f"  Current host: {current_hw['hostname']} "
                            f"({current_hw['cpu_brand']}, "
                            f"{current_hw['cpu_count_logical']} cores)"
                        )
                        print(msg, file=sys.stderr)
                        return 1
                    # Auto-recompute baselines
                    import warnings as _warnings

                    _warnings.warn(
                        f"Dataset baselines computed on {stored_hw.get('hostname', 'unknown')} "
                        f"({stored_hw.get('cpu_brand', '?')}, "
                        f"{stored_hw.get('cpu_count_logical', '?')} cores, "
                        f"{_fmt_ram(stored_hw.get('ram_total_bytes'))}). "
                        f"Current hardware differs. Recomputing baselines...",
                        stacklevel=1,
                    )
                    baselines_recomputed = True
                    # Leave preloaded_baselines as None -> score_estimator_report
                    # will recompute them.
                else:
                    # Hardware matches — use stored baselines
                    budgets_list = list(contest_params.budgets)
                    preloaded_baselines = {}
                    for bi, budget in enumerate(budgets_list):
                        preloaded_baselines[budget] = bundle.baseline_times[bi]

            score_kwargs: dict[str, Any] = {
                "n_circuits": n_circuits,
                "n_samples": n_samples,
                "contest_params": contest_params,
                "entrypoint": entrypoint,
                "limits": _default_resource_limits(),
                "profile": bool(args.profile),
                "detail": str(args.detail),
            }
            if preloaded_circuits is not None:
                score_kwargs["circuits"] = preloaded_circuits
            if preloaded_means is not None:
                score_kwargs["ground_truth_means"] = preloaded_means
            if preloaded_baselines is not None:
                score_kwargs["baseline_times_by_budget"] = preloaded_baselines

            if json_output:
                report = score_estimator_report(runner, **score_kwargs)
                report["mode"] = "agent"
                # Inject dataset traceability
                if dataset_path is not None:
                    report.setdefault("run_config", {})["dataset"] = {
                        "path": str(Path(dataset_path).resolve()),
                        "sha256": dataset_file_hash(dataset_path),
                        "seed": ds_meta.get("seed"),
                        "n_circuits": ds_meta.get("n_circuits"),
                        "n_samples": ds_meta.get("n_samples"),
                        "baselines_recomputed": baselines_recomputed,
                    }
                output = render_agent_report(report)
            else:
                metadata = resolve_estimator_class_metadata(
                    entrypoint.file_path, class_name=entrypoint.class_name
                )
                pre_report = _pre_run_report(
                    n_circuits=n_circuits,
                    n_samples=n_samples,
                    contest_params=contest_params,
                    profile=bool(args.profile),
                    detail=str(args.detail),
                    estimator_class=metadata.class_name,
                    estimator_path=str(entrypoint.file_path),
                )
                total_units = len(contest_params.budgets) * n_circuits
                if rich_tqdm is None:
                    _print_human_startup(
                        pre_report,
                        estimator_class=metadata.class_name,
                        estimator_path=str(entrypoint.file_path),
                    )
                    with _progress_callback(total_units) as progress_cb:
                        score_kwargs["progress"] = progress_cb
                        score_kwargs["sampling_progress"] = progress_cb
                        report = score_estimator_report(runner, **score_kwargs)
                else:
                    _print_human_header_and_hints()
                    with _live_top_pane_session(pre_report, total_units) as live_session:
                        score_kwargs["progress"] = live_session.on_progress
                        score_kwargs["sampling_progress"] = live_session.on_progress
                        report = score_estimator_report(runner, **score_kwargs)
                        run_meta = report.get("run_meta")
                        if isinstance(run_meta, dict):
                            live_session.update_run_meta(run_meta)
                report["mode"] = "human"
                # Inject dataset traceability for human mode too
                if dataset_path is not None:
                    report.setdefault("run_config", {})["dataset"] = {
                        "path": str(Path(dataset_path).resolve()),
                        "sha256": dataset_file_hash(dataset_path),
                        "seed": ds_meta.get("seed"),
                        "n_circuits": ds_meta.get("n_circuits"),
                        "n_samples": ds_meta.get("n_samples"),
                        "baselines_recomputed": baselines_recomputed,
                    }
                try:
                    output = render_human_results(
                        report, show_diagnostic_plots=bool(args.show_diagnostic_plots)
                    )
                except Exception as exc:
                    print(
                        f"Rich dashboard unavailable ({exc}); falling back to plain-text report.",
                        file=sys.stderr,
                    )
                    output = _render_plain_text_report(report)
            print(output, end="" if output.endswith("\n") else "\n")
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
                print(f"Packaged submission: {artifact_path}")
            return 0

        raise ValueError(f"Unsupported command: {command}")
    except Exception as exc:  # pragma: no cover - exercised by CLI tests
        stage = exc.stage if isinstance(exc, RunnerError) else command
        payload = _error_payload(exc, include_traceback=debug, stage=stage)
        _print_error(
            payload,
            json_output=json_output,
            debug=debug,
            show_inprocess_hint=(command == "run" and getattr(args, "runner", None) == "subprocess"),
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    """Dispatch participant subcommands."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    return _main_participant(args_list)


def _print_error(
    payload: dict[str, Any],
    *,
    json_output: bool,
    debug: bool,
    show_inprocess_hint: bool = False,
) -> None:
    if json_output:
        print(json.dumps(payload, indent=2))
        return
    error = payload["error"]
    print(f"Error [{error['stage']}:{error['code']}]: {error['message']}")
    if debug and "traceback" in error:
        print(error["traceback"])
    elif not debug:
        print("Use --debug to include a traceback.")
    if show_inprocess_hint:
        print("Tip: For estimator-level tracebacks, rerun with --runner inprocess --debug.")


def _error_payload(
    exc: Exception,
    *,
    include_traceback: bool,
    stage: str = "scoring",
) -> dict[str, Any]:
    """Build stable error payload shape for human/JSON mode outputs."""
    message = str(exc) or exc.__class__.__name__
    error: dict[str, Any] = {
        "stage": stage,
        "code": _error_code(exc, message),
        "message": message,
    }
    if isinstance(exc, RunnerError) and exc.detail.details:
        error["details"] = exc.detail.details
    if include_traceback:
        error["traceback"] = traceback.format_exc()
    return {"ok": False, "error": error}


def _error_code(exc: Exception, message: str) -> str:
    """Map common failures to stable error codes."""
    if isinstance(exc, RunnerError):
        return exc.detail.code
    lowered = message.lower()
    if isinstance(exc, ValueError):
        if "iterator" in lowered:
            return "ESTIMATOR_STREAM_NOT_ITERABLE"
        if "more than max_depth rows" in lowered:
            return "ESTIMATOR_STREAM_TOO_MANY_ROWS"
        if "exactly max_depth rows" in lowered:
            return "ESTIMATOR_STREAM_TOO_FEW_ROWS"
        if "must have shape" in lowered:
            return "ESTIMATOR_STREAM_BAD_ROW_SHAPE"
        if "finite" in lowered:
            return "ESTIMATOR_STREAM_NON_FINITE_ROW"
        return "SCORING_VALIDATION_ERROR"
    return "SCORING_RUNTIME_ERROR"
