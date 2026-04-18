from __future__ import annotations

from whestbench.presentation.adapters import (
    build_create_dataset_presentation,
    build_init_presentation,
    build_package_presentation,
    build_smoke_test_presentation,
    build_validate_presentation,
)
from whestbench.presentation.models import ChecklistSection, KeyValueSection, StepItem, StepsSection


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
