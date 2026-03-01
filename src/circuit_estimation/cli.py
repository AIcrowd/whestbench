"""CLI and convenience entrypoints for local scoring runs."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit, Layer
from .estimators import CombinedEstimator
from .loader import load_estimator_from_path
from .packaging import package_submission
from .reporting import render_agent_report, render_human_report
from .runner import (
    EstimatorEntrypoint,
    InProcessRunner,
    ResourceLimits,
    RunnerError,
    SubprocessRunner,
)
from .scoring import ContestParams, score_estimator_report, score_submission_report
from .sdk import SetupContext
from .streaming import validate_depth_row

_DEFAULT_ESTIMATOR = CombinedEstimator()
_PARTICIPANT_COMMANDS = frozenset({"init", "validate", "run", "package"})


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
    """Run default scenario and return score-only compatibility output.

    When ``profile`` is true, this mirrors legacy behavior by returning
    ``(score, profile_calls)`` instead of just the numeric score.
    """
    report = run_default_report(profile=profile, detail="raw")
    score = float(report["results"]["final_score"])
    if profile:
        return score, list(report.get("profile_calls", []))
    return score


def run_default_report(*, profile: bool = False, detail: str = "raw") -> dict[str, Any]:
    """Run the default local evaluator scenario and return report payload."""
    return score_estimator_report(
        _DEFAULT_ESTIMATOR.predict,
        n_circuits=10,
        n_samples=10000,
        contest_params=_default_contest_params(),
        profile=profile,
        detail=detail,
    )


def _build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local circuit-estimator scoring.")
    parser.add_argument(
        "--agent-mode",
        action="store_true",
        help="Emit pretty JSON only for machine consumers. Default output is the human dashboard.",
    )
    parser.add_argument(
        "--detail",
        choices=("raw", "full"),
        default="raw",
        help="Report detail level. Use `full` for extra derived metrics.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit per-call profiling diagnostics (wall, cpu, rss, peak_rss).",
    )
    parser.add_argument(
        "--show-diagnostic-plots",
        action="store_true",
        help="Include plot panes for budget/layer/profile diagnostics in human mode.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show traceback details on failures.",
    )
    return parser


def _main_legacy(argv: list[str]) -> int:
    args = _build_legacy_parser().parse_args(argv)
    mode = "agent" if args.agent_mode else "human"
    try:
        report = run_default_report(profile=args.profile, detail=args.detail)
        report["mode"] = mode
        if mode == "agent":
            output = render_agent_report(report)
            print(output, end="" if output.endswith("\n") else "\n")
            return 0

        if _supports_textual_dashboard():
            try:
                if _launch_textual_dashboard(report):
                    return 0
            except Exception as exc:
                print(f"Textual UI unavailable ({exc}); falling back to static report.", file=sys.stderr)
        else:
            print("Textual UI unavailable; falling back to static report.", file=sys.stderr)

        output = render_human_report(report, show_diagnostic_plots=args.show_diagnostic_plots)
        print(output, end="" if output.endswith("\n") else "\n")
        return 0
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        payload = _error_payload(exc, include_traceback=args.debug, stage="scoring")
        _print_error(payload, agent_mode=bool(args.agent_mode), debug=bool(args.debug))
        return 1


def _supports_textual_dashboard() -> bool:
    """Return true when textual dashboard dependencies are importable."""
    try:
        from .textual_dashboard.app import DashboardApp
    except Exception:
        return False
    _ = DashboardApp
    return True


def _launch_textual_dashboard(report: dict[str, Any]) -> bool:
    """Launch the interactive textual dashboard when attached to a TTY."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    from .textual_dashboard.app import DashboardApp

    app = DashboardApp(report=report)
    app.run()
    return True


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

    init_parser = subparsers.add_parser("init", help="Create starter estimator files.")
    init_parser.add_argument("path", nargs="?", default=".")
    init_parser.add_argument("--agent-mode", action="store_true")
    init_parser.add_argument("--debug", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate estimator contract.")
    validate_parser.add_argument(
        "--estimator",
        required=True,
        help="Path to estimator.py (see examples/estimators/ for starter files).",
    )
    validate_parser.add_argument("--class", dest="class_name")
    validate_parser.add_argument("--agent-mode", action="store_true")
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
    run_parser.add_argument("--agent-mode", action="store_true")
    run_parser.add_argument("--debug", action="store_true")

    package_parser = subparsers.add_parser("package", help="Package submission artifact.")
    package_parser.add_argument("--estimator", required=True)
    package_parser.add_argument("--class", dest="class_name")
    package_parser.add_argument("--requirements")
    package_parser.add_argument("--submission-metadata")
    package_parser.add_argument("--approach")
    package_parser.add_argument("--output")
    package_parser.add_argument("--agent-mode", action="store_true")
    package_parser.add_argument("--debug", action="store_true")

    return parser


def _main_participant(argv: list[str]) -> int:
    args = _build_participant_parser().parse_args(argv)
    command = str(args.command)
    agent_mode = bool(getattr(args, "agent_mode", False))
    debug = bool(getattr(args, "debug", False))

    try:
        if command == "init":
            created = _write_init_template(Path(args.path).resolve())
            payload = {"ok": True, "created": created}
            if agent_mode:
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
            if agent_mode:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"Validation passed: class={payload['class_name']} "
                    f"shape={tuple(payload['output_shape'])}"
                )
            return 0

        if command == "run":
            runner = InProcessRunner() if args.runner == "inprocess" else SubprocessRunner()
            report = score_submission_report(
                runner,
                EstimatorEntrypoint(
                    file_path=Path(args.estimator).resolve(),
                    class_name=args.class_name,
                ),
                n_circuits=int(args.n_circuits),
                n_samples=int(args.n_samples),
                contest_params=_default_contest_params(),
                limits=_default_resource_limits(),
                profile=bool(args.profile),
                detail=str(args.detail),
            )
            report["mode"] = "agent" if agent_mode else "human"
            output = (
                render_agent_report(report)
                if agent_mode
                else render_human_report(
                    report, show_diagnostic_plots=bool(args.show_diagnostic_plots)
                )
            )
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
            if agent_mode:
                print(json.dumps(payload, indent=2))
            else:
                print(f"Packaged submission: {artifact_path}")
            return 0

        raise ValueError(f"Unsupported command: {command}")
    except Exception as exc:  # pragma: no cover - exercised by CLI tests
        stage = exc.stage if isinstance(exc, RunnerError) else command
        payload = _error_payload(exc, include_traceback=debug, stage=stage)
        _print_error(payload, agent_mode=agent_mode, debug=debug)
        return 1


def _is_legacy_invocation(argv: list[str]) -> bool:
    if not argv:
        return True
    if argv[0].startswith("-"):
        return True
    return argv[0] not in _PARTICIPANT_COMMANDS


def main(argv: list[str] | None = None) -> int:
    """Dispatch legacy dashboard mode or participant subcommands."""
    args_list = list(argv or [])
    if _is_legacy_invocation(args_list):
        return _main_legacy(args_list)
    return _main_participant(args_list)


def _print_error(payload: dict[str, Any], *, agent_mode: bool, debug: bool) -> None:
    if agent_mode:
        print(json.dumps(payload, indent=2))
        return
    error = payload["error"]
    print(f"Error [{error['stage']}:{error['code']}]: {error['message']}")
    if debug and "traceback" in error:
        print(error["traceback"])
    elif not debug:
        print("Use --debug to include a traceback.")


def _error_payload(
    exc: Exception,
    *,
    include_traceback: bool,
    stage: str = "scoring",
) -> dict[str, Any]:
    """Build stable error payload shape for human/agent mode outputs."""
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
