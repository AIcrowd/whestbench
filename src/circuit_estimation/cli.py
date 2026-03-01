"""CLI and convenience entrypoints for local scoring runs."""

from __future__ import annotations

import argparse
import json
import traceback as traceback_lib
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np

from .domain import Circuit, Layer
from .estimators import combined_estimator
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


def _default_contest_params() -> ContestParams:
    return ContestParams(
        width=100,
        max_depth=30,
        budgets=[10, 100, 1000, 10000],
        time_tolerance=0.1,
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
        combined_estimator,
        n_circuits=10,
        n_samples=10000,
        contest_params=_default_contest_params(),
        profile=profile,
        detail=detail,
    )


def _main_legacy(argv: list[str]) -> int:
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
    args = parser.parse_args(argv)

    report = run_default_report(profile=args.profile, detail=args.detail)
    mode = "agent" if args.agent_mode else "human"
    report["mode"] = mode
    if mode == "agent":
        output = render_agent_report(report)
    else:
        output = render_human_report(report, show_diagnostic_plots=args.show_diagnostic_plots)
    print(output, end="" if output.endswith("\n") else "\n")
    return 0


def _error_payload(
    *,
    stage: str,
    code: str,
    message: str,
    hint: str | None = None,
    details: dict[str, Any] | None = None,
    traceback: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "stage": stage,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "hint": hint or "Check command arguments and estimator contract.",
        },
    }
    if traceback is not None:
        payload["error"]["traceback"] = traceback
    return payload


def _print_error(payload: dict[str, Any], *, agent_mode: bool) -> None:
    if agent_mode:
        print(json.dumps(payload, indent=2))
        return
    error = payload["error"]
    print(f"Error [{payload['stage']}/{error['code']}]: {error['message']}")
    hint = error.get("hint")
    if hint:
        print(f"Hint: {hint}")
    traceback_text = error.get("traceback")
    if traceback_text:
        print(traceback_text)


def _resolve_exception(
    exc: Exception,
    *,
    stage: str,
    debug: bool,
) -> dict[str, Any]:
    traceback_text = traceback_lib.format_exc() if debug else None
    if isinstance(exc, FileNotFoundError):
        return _error_payload(
            stage=stage,
            code="FILE_NOT_FOUND",
            message=str(exc),
            hint="Pass a valid --estimator path.",
            traceback=traceback_text,
        )
    if isinstance(exc, RunnerError):
        return _error_payload(
            stage=exc.stage,
            code=exc.detail.code,
            message=exc.detail.message,
            details=exc.detail.details,
            traceback=traceback_text,
        )
    if isinstance(exc, ValueError):
        return _error_payload(
            stage=stage,
            code="VALIDATION_ERROR",
            message=str(exc),
            traceback=traceback_text,
        )
    return _error_payload(
        stage=stage,
        code="INTERNAL_ERROR",
        message=str(exc),
        traceback=traceback_text,
    )


def _write_init_template(target_dir: Path) -> list[str]:
    created: list[str] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    estimator_file = target_dir / "estimator.py"
    if not estimator_file.exists():
        template_path = Path(__file__).resolve().parent / "templates" / "estimator.py.tmpl"
        estimator_file.write_text(
            template_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        created.append(str(estimator_file))

    requirements_file = target_dir / "requirements.txt"
    if not requirements_file.exists():
        requirements_file.write_text("# Optional custom dependencies\n", encoding="utf-8")
        created.append(str(requirements_file))
    return created


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
    estimator.setup(context)
    predictions = estimator.predict(circuit, 10)
    estimator.teardown()

    if not isinstance(predictions, np.ndarray):
        raise ValueError("Estimator predict must return numpy.ndarray.")
    tensor = np.asarray(predictions, dtype=np.float32)
    if tensor.shape != (circuit.d, circuit.n):
        raise ValueError(
            f"Estimator output shape must be ({circuit.d}, {circuit.n}), got {tensor.shape}."
        )
    if not np.isfinite(tensor).all():
        raise ValueError("Estimator output contains non-finite values.")

    return {
        "ok": True,
        "class_name": metadata.class_name,
        "module_name": metadata.module_name,
        "output_shape": list(tensor.shape),
    }


def _build_participant_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Participant-first circuit-estimation CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create starter estimator files.")
    init_parser.add_argument("path", nargs="?", default=".")
    init_parser.add_argument("--agent-mode", action="store_true")
    init_parser.add_argument("--debug", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate estimator contract.")
    validate_parser.add_argument("--estimator", required=True)
    validate_parser.add_argument("--class", dest="class_name")
    validate_parser.add_argument("--agent-mode", action="store_true")
    validate_parser.add_argument("--debug", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run local evaluation for an estimator.")
    run_parser.add_argument("--estimator", required=True)
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


def _is_legacy_invocation(argv: list[str]) -> bool:
    if not argv:
        return True
    known = {"init", "validate", "run", "package"}
    return argv[0] not in known


def main(argv: list[str] | None = None) -> int:
    """Dispatch legacy dashboard mode or participant subcommands and return exit code."""
    args_list = list(argv or [])
    if _is_legacy_invocation(args_list):
        return _main_legacy(args_list)

    parser = _build_participant_parser()
    args = parser.parse_args(args_list)
    agent_mode = bool(getattr(args, "agent_mode", False))
    debug = bool(getattr(args, "debug", False))

    try:
        if args.command == "init":
            created = _write_init_template(Path(args.path).resolve())
            payload = {"ok": True, "created": created}
            if agent_mode:
                print(json.dumps(payload, indent=2))
            else:
                if created:
                    print("Initialized starter files:")
                    for file in created:
                        print(f"- {file}")
                else:
                    print("Starter files already exist; nothing created.")
            return 0

        if args.command == "validate":
            payload = validate_submission_entrypoint(args.estimator, class_name=args.class_name)
            if agent_mode:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"Validation passed: class={payload['class_name']} shape={tuple(payload['output_shape'])}"
                )
            return 0

        if args.command == "run":
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
                limits=ResourceLimits(
                    setup_timeout_s=5.0,
                    predict_timeout_s=30.0,
                    memory_limit_mb=4096,
                    cpu_time_limit_s=None,
                ),
                profile=bool(args.profile),
                detail=str(args.detail),
            )
            report["mode"] = "agent" if agent_mode else "human"
            if agent_mode:
                output = render_agent_report(report)
            else:
                output = render_human_report(
                    report,
                    show_diagnostic_plots=bool(args.show_diagnostic_plots),
                )
            print(output, end="" if output.endswith("\n") else "\n")
            return 0

        if args.command == "package":
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

        raise ValueError(f"Unsupported command: {args.command}")

    except Exception as exc:
        payload = _resolve_exception(exc, stage=str(args.command), debug=debug)
        _print_error(payload, agent_mode=agent_mode)
        return 1
