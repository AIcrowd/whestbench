from __future__ import annotations

from typing import Any

from .models import (
    ChecklistItem,
    ChecklistSection,
    CommandPresentation,
    ErrorSection,
    KeyValueRow,
    KeyValueSection,
    StepItem,
    StepsSection,
)

_JSON_OUTPUT_TIP = "Use --json for JSON output when calling from automated agents or UIs."
_DIAGNOSTIC_PLOTS_TIP = "Use --show-diagnostic-plots to include diagnostic plot panes."
_SMOKE_TEST_NEXT_STEPS = [
    StepItem(
        purpose="Create starter files you can edit.",
        command="whest init ./my-estimator",
    ),
    StepItem(
        purpose="Validate an Estimator implementation.",
        command="whest validate --estimator ./my-estimator/estimator.py",
    ),
    StepItem(
        purpose="Run local evaluation with isolation.",
        command="whest run --estimator ./my-estimator/estimator.py --runner local",
    ),
    StepItem(
        purpose="Build submission artifacts for AIcrowd.",
        command="whest package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz",
    ),
]


def _display_value(value: Any, *, fallback: str = "n/a") -> str:
    if value in {None, ""}:
        return fallback
    return str(value)


def _status_for_report(report: dict[str, Any]) -> str:
    results = report.get("results")
    if not isinstance(results, dict):
        return "success"
    per_mlp = results.get("per_mlp")
    if not isinstance(per_mlp, list):
        return "success"
    if any(isinstance(entry, dict) and entry.get("error") for entry in per_mlp):
        return "error"
    return "success"


def _base_sections(report: dict[str, Any]) -> list[KeyValueSection]:
    run_config = report.get("run_config")
    if not isinstance(run_config, dict):
        run_config = {}
    run_meta = report.get("run_meta")
    if not isinstance(run_meta, dict):
        run_meta = {}
    results = report.get("results")
    if not isinstance(results, dict):
        results = {}

    return [
        KeyValueSection(
            title="Run Context",
            rows=[
                KeyValueRow("MLPs", _display_value(run_config.get("n_mlps"))),
                KeyValueRow("Width", _display_value(run_config.get("width"))),
                KeyValueRow("Depth", _display_value(run_config.get("depth"))),
                KeyValueRow("FLOP Budget", _display_value(run_config.get("flop_budget"))),
                KeyValueRow("Duration(s)", _display_value(run_meta.get("run_duration_s"))),
                KeyValueRow(
                    "Wall Time Limit",
                    _display_value(run_config.get("wall_time_limit_s"), fallback="unlimited"),
                ),
                KeyValueRow(
                    "Untracked Time Limit",
                    _display_value(run_config.get("untracked_time_limit_s"), fallback="unlimited"),
                ),
            ],
        ),
        KeyValueSection(
            title="Final Score",
            rows=[
                KeyValueRow("Primary Score", _display_value(results.get("primary_score"))),
                KeyValueRow("Secondary Score", _display_value(results.get("secondary_score"))),
            ],
        ),
    ]


def build_run_presentation(report: dict[str, Any], *, debug: bool) -> CommandPresentation:
    del debug
    return CommandPresentation(
        command="run",
        status=_status_for_report(report),
        title="WhestBench Report",
        sections=_base_sections(report),
        epilogue_messages=[_JSON_OUTPUT_TIP, _DIAGNOSTIC_PLOTS_TIP],
    )


def build_smoke_test_presentation(report: dict[str, Any], *, debug: bool) -> CommandPresentation:
    base_doc = build_run_presentation(report, debug=debug)
    return CommandPresentation(
        command="smoke-test",
        status=base_doc.status,
        title=base_doc.title,
        subtitle=base_doc.subtitle,
        sections=[
            *base_doc.sections,
            StepsSection(
                title="Next Steps",
                steps=list(_SMOKE_TEST_NEXT_STEPS),
            ),
        ],
        epilogue_messages=list(base_doc.epilogue_messages),
    )


def build_validate_presentation(payload: dict[str, Any]) -> CommandPresentation:
    raw_checks = payload.get("checks")
    checks = raw_checks if isinstance(raw_checks, list) else []
    return CommandPresentation(
        command="validate",
        status="success",
        title="Validation",
        sections=[
            ChecklistSection(
                title="Checks",
                items=[
                    ChecklistItem(
                        label=str(item.get("name", "")),
                        status=str(item.get("status", "ok")),
                        detail=str(item.get("detail", "")),
                    )
                    for item in checks
                    if isinstance(item, dict)
                ],
            )
        ],
    )


def build_init_presentation(payload: dict[str, Any]) -> CommandPresentation:
    raw_created = payload.get("created")
    created = [str(path) for path in raw_created] if isinstance(raw_created, list) else []

    if created:
        sections: list[KeyValueSection | StepsSection] = [
            StepsSection(title="Created Files", steps=created)
        ]
    else:
        sections = [
            KeyValueSection(
                title="Status",
                rows=[KeyValueRow("Message", "Starter files already exist; nothing created.")],
            )
        ]

    return CommandPresentation(
        command="init",
        status="success",
        title="Starter Files",
        sections=sections,
    )


def build_create_dataset_presentation(payload: dict[str, Any]) -> CommandPresentation:
    return CommandPresentation(
        command="create-dataset",
        status="success",
        title="Dataset Created",
        sections=[
            KeyValueSection(
                title="Dataset",
                rows=[KeyValueRow("Path", _display_value(payload.get("path")))],
            )
        ],
    )


def build_package_presentation(payload: dict[str, Any]) -> CommandPresentation:
    return CommandPresentation(
        command="package",
        status="success",
        title="Packaged Submission",
        sections=[
            KeyValueSection(
                title="Artifact",
                rows=[KeyValueRow("Path", _display_value(payload.get("artifact_path")))],
            )
        ],
    )


def build_error_presentation(
    payload: dict[str, Any],
    *,
    debug: bool,
    show_inprocess_hint: bool,
) -> CommandPresentation:
    error = payload.get("error")
    if not isinstance(error, dict):
        error = {}

    stage = str(error.get("stage") or "scoring")
    code = str(error.get("code") or "SCORING_RUNTIME_ERROR")
    message = str(error.get("message") or "Unknown error")
    details = error.get("details")
    traceback_text = error.get("traceback")

    section = ErrorSection(
        title="Failure Details",
        code=code,
        message=message,
        details=dict(details) if isinstance(details, dict) else {},
        traceback=str(traceback_text) if debug and traceback_text else None,
    )

    epilogue_messages = []
    if not debug:
        epilogue_messages.append("Use --debug to include a traceback.")
    if show_inprocess_hint:
        epilogue_messages.append(
            "Tip: For estimator-level tracebacks, rerun with --runner local --debug."
        )

    return CommandPresentation(
        command=stage,
        status="error",
        title=f"Error [{stage}:{code}]",
        sections=[section],
        epilogue_messages=epilogue_messages,
    )
