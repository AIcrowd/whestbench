from __future__ import annotations

import re

import whestbench.cli as cli
from whestbench.presentation.adapters import (
    build_create_dataset_presentation,
    build_error_presentation,
    build_init_presentation,
    build_package_presentation,
    build_run_presentation,
    build_smoke_test_presentation,
    build_validate_presentation,
)
from whestbench.presentation.render_plain import render_plain_presentation
from whestbench.presentation.render_rich import render_rich_presentation


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _sample_run_report() -> dict[str, object]:
    return {
        "run_meta": {"run_duration_s": 1.0, "host": {}},
        "run_config": {"n_mlps": 1, "width": 4, "depth": 2, "flop_budget": 100},
        "results": {"primary_score": 0.42, "secondary_score": 0.55, "per_mlp": []},
    }


def test_parity_matrix_preserves_settled_information() -> None:
    docs = [
        build_run_presentation(_sample_run_report(), debug=False),
        build_smoke_test_presentation(
            {
                "run_meta": {"run_duration_s": 1.0, "host": {}},
                "run_config": {
                    "n_mlps": 3,
                    "width": 100,
                    "depth": 16,
                    "flop_budget": 10_000_000,
                },
                "results": {"primary_score": 0.42, "secondary_score": 0.55, "per_mlp": []},
            },
            debug=False,
        ),
        build_init_presentation({"ok": True, "created": ["/tmp/demo/estimator.py"]}),
        build_create_dataset_presentation({"ok": True, "path": "/tmp/eval_dataset.npz"}),
        build_package_presentation({"ok": True, "artifact_path": "/tmp/submission.tar.gz"}),
        build_validate_presentation(
            {
                "ok": True,
                "class_name": "Estimator",
                "module_name": "_submission",
                "output_shape": [2, 4],
                "checks": [{"name": "class resolved", "status": "ok", "detail": "Estimator"}],
            }
        ),
        build_error_presentation(
            {
                "ok": False,
                "error": {
                    "stage": "validate",
                    "code": "ESTIMATOR_BAD_SHAPE",
                    "message": "Predictions must have shape (2, 4), got (4, 2).",
                    "details": {"expected_shape": [2, 4], "got_shape": [4, 2]},
                },
            },
            debug=False,
            show_inprocess_hint=False,
        ),
    ]

    for doc in docs:
        plain = render_plain_presentation(doc)
        rich = _strip_ansi(render_rich_presentation(doc))

        required_tokens = [doc.title, *doc.epilogue_messages]
        required_tokens.extend(
            section.title for section in doc.sections if getattr(section, "title", "")
        )

        for token in required_tokens:
            if token:
                assert token in plain
                assert token in rich


def test_run_plain_text_report_keeps_shared_run_epilogues() -> None:
    rendered = cli._render_plain_text_report(_sample_run_report())

    assert "Use --json for JSON output when calling from automated agents or UIs." in rendered
    assert "Use --show-diagnostic-plots to include diagnostic plot panes." in rendered
