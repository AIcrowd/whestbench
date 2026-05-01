from __future__ import annotations

from whestbench.presentation.adapters import (
    build_create_dataset_presentation,
    build_init_presentation,
    build_package_presentation,
    build_profile_presentation,
    build_run_presentation,
    build_smoke_test_presentation,
    build_validate_presentation,
)
from whestbench.presentation.models import (
    BudgetBreakdownSection,
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
    steps = [step for step in next_steps.steps if isinstance(step, StepItem)]
    assert len(steps) == len(next_steps.steps)
    assert [(step.purpose, step.command) for step in steps] == [
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


def test_build_run_presentation_orders_breakdowns_before_final_score() -> None:
    doc = build_run_presentation(
        {
            "run_meta": {"run_duration_s": 1.0, "host": {}},
            "run_config": {"n_mlps": 2, "width": 4, "depth": 3, "flop_budget": 100},
            "results": {
                "primary_score": 0.123,
                "secondary_score": 0.456,
                "per_mlp": [],
                "breakdowns": {
                    "sampling": {
                        "flops_used": 80,
                        "tracked_time_s": 0.02,
                        "flopscope_overhead_time_s": 0.005,
                        "untracked_time_s": 0.01,
                        "by_namespace": {
                            "sampling.sample_layer_statistics": {
                                "flops_used": 80,
                                "tracked_time_s": 0.02,
                                "flopscope_overhead_time_s": 0.005,
                            }
                        },
                    },
                    "estimator": {
                        "flop_budget": 200,
                        "flops_used": 90,
                        "tracked_time_s": 0.03,
                        "flopscope_overhead_time_s": 0.0075,
                        "untracked_time_s": 0.01,
                        "by_namespace": {
                            "estimator.estimator-client": {
                                "flops_used": 90,
                                "tracked_time_s": 0.03,
                                "flopscope_overhead_time_s": 0.0075,
                            }
                        },
                    },
                },
            },
        },
        debug=False,
    )

    titles = [section.title for section in doc.sections]

    assert titles.index("Sampling Budget Breakdown (Ground Truth)") < titles.index(
        "Estimator Budget Breakdown"
    )
    assert titles.index("Estimator Budget Breakdown") < titles.index("Final Score")
    sampling = next(
        section
        for section in doc.sections
        if isinstance(section, BudgetBreakdownSection)
        and section.title == "Sampling Budget Breakdown (Ground Truth)"
    )
    estimator = next(
        section
        for section in doc.sections
        if isinstance(section, BudgetBreakdownSection)
        and section.title == "Estimator Budget Breakdown"
    )
    assert sampling.available is True
    assert estimator.available is True
    assert estimator.gauge is not None


def test_build_run_presentation_restores_main_style_score_and_context_fields() -> None:
    doc = build_run_presentation(
        {
            "run_meta": {
                "run_duration_s": 1.0,
                "host": {
                    "hostname": "example-host",
                    "os": "Darwin",
                    "python_version": "3.13.7",
                },
            },
            "run_config": {
                "estimator_class": "CombinedEstimator",
                "estimator_path": "examples/estimators/combined_estimator.py",
                "n_mlps": 2,
                "width": 4,
                "depth": 3,
                "flop_budget": 100,
            },
            "results": {
                "primary_score": 0.123456789,
                "secondary_score": 0.456789123,
                "per_mlp": [
                    {"mlp_index": 0, "final_mse": 0.1},
                    {"mlp_index": 1, "final_mse": 0.2},
                ],
            },
        },
        debug=False,
    )

    run_context = next(
        section
        for section in doc.sections
        if isinstance(section, KeyValueSection) and section.title == "Run Context"
    )
    score = next(
        section
        for section in doc.sections
        if isinstance(section, TableSection) and section.title == "Final Score"
    )

    assert [row.label for row in run_context.rows][:4] == [
        "Estimator Class [estimator_class]",
        "Estimator Path [estimator_path]",
        "Started [run_started_at_utc]",
        "Finished [run_finished_at_utc]",
    ]
    assert score.columns == ["metric", "value"]
    assert score.rows == [
        ["Primary Score [primary_score]", "0.12345679"],
        ["Secondary Score [secondary_score]", "0.45678912"],
        ["Best MLP Score [best_mlp_score]", "0.10000000"],
        ["Worst MLP Score [worst_mlp_score]", "0.20000000"],
    ]
    assert (
        score.subtitle == "lower MSE is better; primary score = mean across MLPs of final-layer MSE"
    )


def test_build_run_presentation_marks_dataset_sampling_breakdown_as_unavailable() -> None:
    doc = build_run_presentation(
        {
            "run_meta": {"run_duration_s": 1.0, "host": {}},
            "run_config": {
                "n_mlps": 2,
                "width": 4,
                "depth": 3,
                "flop_budget": 100,
                "dataset": {"path": "/tmp/eval.npz", "sha256": "abc123"},
            },
            "results": {
                "primary_score": 0.123,
                "secondary_score": 0.456,
                "per_mlp": [],
                "breakdowns": {"sampling": None, "estimator": None},
            },
        },
        debug=False,
    )

    sampling = next(
        section
        for section in doc.sections
        if isinstance(section, BudgetBreakdownSection)
        and section.title == "Sampling Budget Breakdown (Ground Truth)"
    )

    assert sampling.available is False
    assert sampling.unavailable_message is not None
    assert "recreate the dataset" in sampling.unavailable_message.lower()


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


def test_build_profile_presentation_includes_correctness_and_timing_rows() -> None:
    doc = build_profile_presentation(
        {
            "hardware": {"os": "Darwin", "machine": "arm64", "python_version": "3.14.3"},
            "correctness": [{"backend": "flopscope", "passed": True, "error": ""}],
            "timing": [
                {
                    "backend": "flopscope",
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
    assert detail.rows == [["flopscope", "256×4×10k", "0.0444s", "0.1135s"]]
