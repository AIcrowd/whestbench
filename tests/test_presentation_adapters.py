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
                        "flopscope_backend_time_s": 0.02,
                        "flopscope_overhead_time_s": 0.005,
                        "residual_wall_time_s": 0.01,
                        "by_namespace": {
                            "sampling.sample_layer_statistics": {
                                "flops_used": 80,
                                "flopscope_backend_time_s": 0.02,
                                "flopscope_overhead_time_s": 0.005,
                            }
                        },
                    },
                    "estimator": {
                        "flop_budget": 200,
                        "flops_used": 90,
                        "flopscope_backend_time_s": 0.03,
                        "flopscope_overhead_time_s": 0.0075,
                        "residual_wall_time_s": 0.01,
                        "by_namespace": {
                            "estimator.estimator-client": {
                                "flops_used": 90,
                                "flopscope_backend_time_s": 0.03,
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
                "adjusted_final_layer_mse": 0.123456789,
                "final_layer_mse": 0.115,
                "all_layers_mse": 0.456789123,
                "best_mlp_adjusted_final_layer_mse": 0.1,
                "worst_mlp_adjusted_final_layer_mse": 0.2,
                "mean_score_multiplier": 0.9,
                "mean_compute_utilization": 0.5,
                "n_failed_mlps": 0,
                "per_mlp": [
                    {"mlp_index": 0, "adjusted_final_layer_mse": 0.1},
                    {"mlp_index": 1, "adjusted_final_layer_mse": 0.2},
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
    assert score.columns == ["metric", "value", "note"]
    assert score.rows == [
        ["Adjusted Final-Layer MSE [adjusted_final_layer_mse]", "1.23e-01", "← primary score"],
        ["Raw Final-Layer MSE [final_layer_mse]", "1.15e-01", ""],
        ["All-Layers MSE [all_layers_mse]", "4.57e-01", ""],
        ["Best MLP [best_mlp_adjusted_final_layer_mse]", "1.00e-01", ""],
        ["Worst MLP [worst_mlp_adjusted_final_layer_mse]", "2.00e-01", ""],
        ["Mean Score Multiplier [mean_score_multiplier]", "0.90000000", ""],
        ["Mean Compute Utilization [mean_compute_utilization]", "0.50000000", ""],
        ["Failed MLPs [n_failed_mlps]", "0 of 2", ""],
    ]
    assert score.subtitle is not None
    assert "max(0.1, C_m/B_m)" in score.subtitle
    assert "lower is better" in score.subtitle


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


def test_score_section_uses_new_score_key_names_and_subtitle():
    from whestbench.presentation.adapters import _score_section

    report = {
        "results": {
            "adjusted_final_layer_mse": 0.245,
            "final_layer_mse": 0.220,
            "all_layers_mse": 0.178,
            "best_mlp_adjusted_final_layer_mse": 0.00001956,
            "worst_mlp_adjusted_final_layer_mse": 0.73648548,
            "mean_score_multiplier": 0.78,
            "mean_compute_utilization": 0.62,
            "n_failed_mlps": 1,
            "per_mlp": [
                {"adjusted_final_layer_mse": 0.00001956},
                {"adjusted_final_layer_mse": 0.5},
                {"adjusted_final_layer_mse": 0.73648548},
            ],
        }
    }
    section = _score_section(report)
    # Subtitle describes new scoring formula
    assert section.subtitle is not None
    assert "s_m" in section.subtitle or "max(0.1" in section.subtitle
    assert "final_layer_mse" in section.subtitle or "C_m" in section.subtitle

    # Row labels reference the new key codes
    metric_labels = [row[0] for row in section.rows]
    joined = " | ".join(metric_labels)
    assert "adjusted_final_layer_mse" in joined
    assert "final_layer_mse" in joined
    assert "all_layers_mse" in joined
    assert "mean_score_multiplier" in joined
    assert "mean_compute_utilization" in joined
    assert "n_failed_mlps" in joined
    assert "best_mlp_adjusted_final_layer_mse" in joined
    assert "worst_mlp_adjusted_final_layer_mse" in joined


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


def test_score_section_handles_null_n_failed_and_per_mlp():
    """A report with null n_failed_mlps or null per_mlp must not crash _score_section."""
    from whestbench.presentation.adapters import _score_section

    report = {
        "results": {
            "adjusted_final_layer_mse": 0.245,
            "final_layer_mse": 0.220,
            "all_layers_mse": 0.178,
            "best_mlp_adjusted_final_layer_mse": 0.0,
            "worst_mlp_adjusted_final_layer_mse": 1.0,
            "mean_score_multiplier": 0.5,
            "mean_compute_utilization": 0.0,
            "n_failed_mlps": None,
            "per_mlp": None,
        }
    }
    section = _score_section(report)
    # Find the Failed MLPs row
    failed_row = next(r for r in section.rows if r[0] == "Failed MLPs [n_failed_mlps]")
    assert failed_row[1] == "0 of 0"
