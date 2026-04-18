from __future__ import annotations

from whestbench.presentation.adapters import (
    build_create_dataset_presentation,
    build_init_presentation,
    build_package_presentation,
    build_profile_presentation,
    build_smoke_test_presentation,
    build_validate_presentation,
    build_visualizer_error_presentation,
    build_visualizer_ready_presentation,
)
from whestbench.presentation.models import (
    ChecklistSection,
    KeyValueSection,
    StepItem,
    StepsSection,
    TableSection,
)


def test_build_smoke_test_presentation_includes_structured_next_steps() -> None:
    report = {
        "run_meta": {"run_duration_s": 1.0, "host": {}},
        "run_config": {"n_mlps": 3, "width": 100, "depth": 16, "flop_budget": 10_000_000},
        "results": {"primary_score": 0.42, "secondary_score": 0.55, "per_mlp": []},
    }

    doc = build_smoke_test_presentation(report, debug=False)

    next_steps = next(
        section
        for section in doc.sections
        if isinstance(section, StepsSection) and section.title == "Next Steps"
    )
    assert all(isinstance(step, StepItem) for step in next_steps.steps)
    assert [(step.purpose, step.command) for step in next_steps.steps] == [
        ("Create starter files you can edit.", "whest init ./my-estimator"),
        (
            "Validate an Estimator implementation.",
            "whest validate --estimator ./my-estimator/estimator.py",
        ),
        (
            "Run local evaluation with isolation.",
            "whest run --estimator ./my-estimator/estimator.py --runner local",
        ),
        (
            "Build submission artifacts for AIcrowd.",
            "whest package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz",
        ),
    ]


def test_build_validate_presentation_includes_structured_checklist() -> None:
    doc = build_validate_presentation(
        {
            "ok": True,
            "class_name": "Estimator",
            "module_name": "_submission",
            "output_shape": [2, 4],
            "checks": [
                {"name": "class resolved", "status": "ok", "detail": "Estimator"},
                {"name": "predict() returned shape", "status": "ok", "detail": "(2, 4)"},
            ],
        }
    )

    checklist = next(
        section
        for section in doc.sections
        if isinstance(section, ChecklistSection) and section.title == "Checks"
    )

    assert doc.command == "validate"
    assert doc.status == "success"
    assert doc.title == "Validation"
    assert [(item.label, item.status, item.detail) for item in checklist.items] == [
        ("class resolved", "ok", "Estimator"),
        ("predict() returned shape", "ok", "(2, 4)"),
    ]


def test_build_init_presentation_includes_created_files() -> None:
    doc = build_init_presentation(
        {"ok": True, "created": ["/tmp/demo/estimator.py", "/tmp/demo/requirements.txt"]}
    )

    created_files = next(
        section
        for section in doc.sections
        if isinstance(section, StepsSection) and section.title == "Created Files"
    )

    assert doc.command == "init"
    assert doc.status == "success"
    assert doc.title == "Starter Files"
    assert list(created_files.steps) == ["/tmp/demo/estimator.py", "/tmp/demo/requirements.txt"]


def test_build_init_presentation_includes_noop_status_when_no_files_created() -> None:
    doc = build_init_presentation({"ok": True, "created": []})

    status = next(
        section
        for section in doc.sections
        if isinstance(section, KeyValueSection) and section.title == "Status"
    )

    assert doc.command == "init"
    assert doc.status == "success"
    assert doc.title == "Starter Files"
    assert [(row.label, row.value) for row in status.rows] == [
        ("Message", "Starter files already exist; nothing created.")
    ]


def test_build_create_dataset_presentation_includes_dataset_path() -> None:
    doc = build_create_dataset_presentation({"ok": True, "path": "/tmp/eval_dataset.npz"})

    dataset = next(
        section
        for section in doc.sections
        if isinstance(section, KeyValueSection) and section.title == "Dataset"
    )

    assert doc.command == "create-dataset"
    assert doc.status == "success"
    assert doc.title == "Dataset Created"
    assert [(row.label, row.value) for row in dataset.rows] == [("Path", "/tmp/eval_dataset.npz")]


def test_build_package_presentation_includes_artifact_path() -> None:
    doc = build_package_presentation({"ok": True, "artifact_path": "/tmp/submission.tar.gz"})

    artifact = next(
        section
        for section in doc.sections
        if isinstance(section, KeyValueSection) and section.title == "Artifact"
    )

    assert doc.command == "package"
    assert doc.status == "success"
    assert doc.title == "Packaged Submission"
    assert [(row.label, row.value) for row in artifact.rows] == [("Path", "/tmp/submission.tar.gz")]


def test_build_visualizer_ready_presentation_contains_url() -> None:
    doc = build_visualizer_ready_presentation(
        {
            "url": "http://127.0.0.1:4173/",
            "host": "127.0.0.1",
            "port": 4173,
            "no_open": True,
            "ran_npm_ci": True,
        }
    )

    ready = next(
        section
        for section in doc.sections
        if isinstance(section, KeyValueSection) and section.title == "Ready"
    )

    assert doc.command == "visualizer"
    assert doc.status == "success"
    assert doc.title == "WhestBench Explorer"
    assert [(row.label, row.value) for row in ready.rows] == [
        ("URL", "http://127.0.0.1:4173/"),
        ("Host", "127.0.0.1"),
        ("Port", "4173"),
    ]
    assert doc.epilogue_messages == [
        "Browser auto-open disabled.",
        "Dependencies were installed with npm ci before launch.",
    ]


def test_build_visualizer_error_presentation_wraps_message() -> None:
    doc = build_visualizer_error_presentation("Missing Prerequisite", "node is not installed.")

    failure = next(
        section
        for section in doc.sections
        if isinstance(section, KeyValueSection) and section.title == "Failure"
    )

    assert doc.command == "visualizer"
    assert doc.status == "error"
    assert doc.title == "Missing Prerequisite"
    assert [(row.label, row.value) for row in failure.rows] == [
        ("Message", "node is not installed.")
    ]


def test_build_profile_presentation_includes_correctness_and_timing_rows() -> None:
    doc = build_profile_presentation(
        {
            "hardware": {"os": "Darwin", "machine": "arm64", "python_version": "3.14.3"},
            "correctness": [{"backend": "whest", "passed": True, "error": ""}],
            "timing": [
                {
                    "backend": "whest",
                    "dims": "256×4×10k",
                    "run_mlp": "0.0444s",
                    "sample_layer_statistics": "0.1135s",
                }
            ],
            "verbose": False,
        }
    )

    assert doc.title == "Simulation Profile"
    assert any(section.title == "Correctness" for section in doc.sections)

    detail = next(
        section
        for section in doc.sections
        if isinstance(section, TableSection) and section.title == "Detail"
    )
    assert detail.columns == ["Backend", "Dims", "run_mlp", "sample_layer_statistics"]
    assert detail.rows == [["whest", "256×4×10k", "0.0444s", "0.1135s"]]
