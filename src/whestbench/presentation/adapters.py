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
    TableSection,
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


def _non_empty_messages(messages: list[str]) -> list[str]:
    return [message for message in messages if message]


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
        epilogue_messages=_non_empty_messages([_JSON_OUTPUT_TIP, _DIAGNOSTIC_PLOTS_TIP]),
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
        epilogue_messages=_non_empty_messages(list(base_doc.epilogue_messages)),
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


def build_visualizer_ready_presentation(payload: dict[str, Any]) -> CommandPresentation:
    host = str(payload.get("host") or "localhost")
    port = payload.get("port")
    port_text = str(port) if port not in {None, ""} else "5173"
    url = str(payload.get("url") or "")
    if not url:
        browser_host = "localhost" if host == "0.0.0.0" else host
        url = f"http://{browser_host}:{port_text}/"

    return CommandPresentation(
        command="visualizer",
        status="success",
        title="WhestBench Explorer",
        sections=[
            KeyValueSection(
                title="Ready",
                rows=[
                    KeyValueRow("URL", url),
                    KeyValueRow("Host", host),
                    KeyValueRow("Port", port_text),
                ],
            )
        ],
        epilogue_messages=_non_empty_messages(
            [
                "Browser auto-open disabled." if payload.get("no_open") else "",
                (
                    "Dependencies were installed with npm ci before launch."
                    if payload.get("ran_npm_ci")
                    else ""
                ),
            ]
        ),
    )


def build_visualizer_error_presentation(
    title: str,
    code: str,
    message: str,
    *,
    details: str | None = None,
    next_steps: list[str] | None = None,
) -> CommandPresentation:
    sections: list[ErrorSection | StepsSection] = [
        ErrorSection(
            title="Failure Details",
            code=code,
            message=message,
            details={"stderr": details} if details else {},
        )
    ]
    if next_steps:
        sections.append(StepsSection(title="Next Steps", steps=list(next_steps)))

    return CommandPresentation(
        command="visualizer",
        status="error",
        title=title,
        sections=sections,
    )


def build_profile_presentation(payload: dict[str, Any]) -> CommandPresentation:
    hardware = payload.get("hardware")
    correctness = payload.get("correctness")
    timing = payload.get("timing")
    verbose = bool(payload.get("verbose"))

    sections: list[KeyValueSection | TableSection] = []

    hardware_rows: list[KeyValueRow] = []
    if isinstance(hardware, dict):
        for key, label in (
            ("os", "OS"),
            ("machine", "Architecture"),
            ("cpu_count_physical", "Physical Cores"),
            ("cpu_count_logical", "Logical Cores"),
            ("ram_total_bytes", "RAM"),
            ("python_version", "Python"),
            ("numpy_version", "NumPy"),
        ):
            value = hardware.get(key)
            if value is None:
                continue
            if key == "ram_total_bytes":
                value = f"{float(value) / (1024**3):.1f} GB"
            hardware_rows.append(KeyValueRow(label, str(value)))
    if hardware_rows:
        sections.append(KeyValueSection(title="Hardware", rows=hardware_rows))

    correctness_rows: list[KeyValueRow] = []
    passed_any = False
    if isinstance(correctness, list):
        for item in correctness:
            if not isinstance(item, dict):
                continue
            passed = bool(item.get("passed"))
            passed_any = passed_any or passed
            correctness_rows.append(
                KeyValueRow(
                    str(item.get("backend", "")),
                    "PASS" if passed else str(item.get("error") or "FAIL"),
                )
            )
    if correctness_rows:
        sections.append(KeyValueSection(title="Correctness", rows=correctness_rows))

    detail_rows: list[list[str]] = []
    if isinstance(timing, list):
        for row in timing:
            if not isinstance(row, dict):
                continue
            detail_rows.append(
                [
                    str(row.get("backend", "")),
                    str(row.get("dims", "")),
                    str(row.get("run_mlp", "")),
                    str(row.get("sample_layer_statistics", "")),
                ]
            )
    if detail_rows:
        sections.append(
            TableSection(
                title="Detail",
                columns=["Backend", "Dims", "run_mlp", "sample_layer_statistics"],
                rows=detail_rows,
            )
        )

    return CommandPresentation(
        command="profile-simulation",
        status="success" if passed_any else "error",
        title="Simulation Profile",
        sections=sections,
        epilogue_messages=_non_empty_messages(
            [
                (
                    "Use --verbose for full timing tables with raw times"
                    if not verbose and passed_any
                    else ""
                ),
                "Use --verbose for error details." if not verbose and not passed_any else "",
            ]
        ),
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

    return CommandPresentation(
        command=stage,
        status="error",
        title=f"Error [{stage}:{code}]",
        sections=[section],
        epilogue_messages=_non_empty_messages(
            [
                "Use --debug to include a traceback." if not debug else "",
                (
                    "Tip: For estimator-level tracebacks, rerun with --runner local --debug."
                    if show_inprocess_hint
                    else ""
                ),
            ]
        ),
    )
