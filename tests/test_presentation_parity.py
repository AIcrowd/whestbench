from __future__ import annotations

import re
from contextlib import contextmanager

import whestbench.cli as cli
from whestbench.presentation.adapters import (
    build_create_dataset_presentation,
    build_error_presentation,
    build_init_presentation,
    build_package_presentation,
    build_profile_presentation,
    build_run_presentation,
    build_smoke_test_presentation,
    build_validate_presentation,
    build_visualizer_error_presentation,
    build_visualizer_ready_presentation,
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


@contextmanager
def _fake_live_session(*_args: object, **_kwargs: object):
    class _Session:
        def on_progress(self, _event: dict[str, object]) -> None:
            return None

        def update_run_meta(self, _run_meta: dict[str, object]) -> None:
            return None

    yield _Session()


def test_parity_matrix_preserves_settled_information() -> None:
    cases = [
        (
            build_run_presentation(_sample_run_report(), debug=False),
            [
                "WhestBench Report",
                "Run Context",
                "MLPs",
                "1",
                "Primary Score",
                "0.42",
                "Use --json for JSON output when calling from automated agents or UIs.",
                "Use --show-diagnostic-plots to include diagnostic plot panes.",
            ],
        ),
        (
            build_smoke_test_presentation(
                {
                    "run_meta": {"run_duration_s": 1.0, "host": {}},
                    "run_config": {
                        "n_mlps": 3,
                        "width": 100,
                        "depth": 16,
                        "flop_budget": 10_000_000,
                    },
                    "results": {
                        "primary_score": 0.42,
                        "secondary_score": 0.55,
                        "per_mlp": [],
                    },
                },
                debug=False,
            ),
            [
                "WhestBench Report",
                "Next Steps",
                "Create starter files you can edit.",
                "whest init ./my-estimator",
                "Use --json for JSON output when calling from automated agents or UIs.",
            ],
        ),
        (
            build_init_presentation({"ok": True, "created": ["/tmp/demo/estimator.py"]}),
            ["Starter Files", "Created Files", "/tmp/demo/estimator.py"],
        ),
        (
            build_create_dataset_presentation({"ok": True, "path": "/tmp/eval_dataset.npz"}),
            ["Dataset Created", "Dataset", "/tmp/eval_dataset.npz"],
        ),
        (
            build_package_presentation({"ok": True, "artifact_path": "/tmp/submission.tar.gz"}),
            ["Packaged Submission", "Artifact", "/tmp/submission.tar.gz"],
        ),
        (
            build_validate_presentation(
                {
                    "ok": True,
                    "class_name": "Estimator",
                    "module_name": "_submission",
                    "output_shape": [2, 4],
                    "checks": [{"name": "class resolved", "status": "ok", "detail": "Estimator"}],
                }
            ),
            ["Validation", "Checks", "class resolved", "Estimator"],
        ),
        (
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
            [
                "Error [validate:ESTIMATOR_BAD_SHAPE]",
                "Predictions must have shape (2, 4), got (4, 2).",
                "Expected shape: [2, 4]",
                "Got shape: [4, 2]",
                "Use --debug to include a traceback.",
            ],
        ),
        (
            build_visualizer_ready_presentation(
                {
                    "url": "http://127.0.0.1:4173/",
                    "host": "127.0.0.1",
                    "port": 4173,
                    "no_open": True,
                    "ran_npm_ci": True,
                }
            ),
            [
                "WhestBench Explorer",
                "Ready",
                "http://127.0.0.1:4173/",
                "127.0.0.1",
                "4173",
                "Browser auto-open disabled.",
                "Dependencies were installed with npm ci before launch.",
            ],
        ),
        (
            build_visualizer_error_presentation(
                "Missing Prerequisite",
                "VISUALIZER_NODE_MISSING",
                "node is not installed.",
                next_steps=[
                    "macOS: brew install node",
                    "Ubuntu/Debian: sudo apt install nodejs npm",
                ],
            ),
            [
                "Missing Prerequisite",
                "node is not installed.",
                "Next Steps",
                "macOS: brew install node",
                "Ubuntu/Debian: sudo apt install nodejs npm",
            ],
        ),
        (
            build_profile_presentation(
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
            ),
            [
                "Simulation Profile",
                "Hardware",
                "Darwin",
                "Correctness",
                "PASS",
                "Detail",
                "256×4×10k",
                "Use --verbose for full timing tables with raw times",
            ],
        ),
    ]

    for doc, required_tokens in cases:
        plain = render_plain_presentation(doc)
        rich = _strip_ansi(render_rich_presentation(doc))

        for token in required_tokens:
            if token:
                assert token in plain
                assert token in rich


def test_run_rich_fallback_keeps_shared_run_epilogues_once(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )
    monkeypatch.setattr(cli, "_run_estimator_with_runner", lambda *_a, **_k: _sample_run_report())
    monkeypatch.setattr(cli, "_print_human_header_and_hints", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_live_top_pane_session", _fake_live_session, raising=True)

    def _fail_render(*_args, **_kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(cli, "render_human_results", _fail_render, raising=False)

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "inprocess"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Rich dashboard unavailable (render failed)" in captured.err
    assert (
        captured.out.count("Use --json for JSON output when calling from automated agents or UIs.")
        == 1
    )
    assert captured.out.count("Use --show-diagnostic-plots to include diagnostic plot panes.") == 1


def test_run_rich_fallback_omits_diagnostic_plots_hint_when_already_enabled(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )
    monkeypatch.setattr(cli, "_run_estimator_with_runner", lambda *_a, **_k: _sample_run_report())
    monkeypatch.setattr(cli, "_print_human_header_and_hints", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_live_top_pane_session", _fake_live_session, raising=True)

    def _fail_render(*_args, **_kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(cli, "render_human_results", _fail_render, raising=False)

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            "estimator.py",
            "--runner",
            "inprocess",
            "--show-diagnostic-plots",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Rich dashboard unavailable (render failed)" in captured.err
    assert (
        captured.out.count("Use --json for JSON output when calling from automated agents or UIs.")
        == 1
    )
    assert "Use --show-diagnostic-plots to include diagnostic plot panes." not in captured.out
