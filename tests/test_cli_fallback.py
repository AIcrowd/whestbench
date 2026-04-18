from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import whestbench.cli as cli


def _sample_report() -> dict:
    return {
        "mode": "human",
        "run_meta": {
            "run_duration_s": 1.25,
        },
        "run_config": {
            "n_mlps": 2,
            "width": 4,
            "depth": 3,
            "flop_budget": 40000,
        },
        "results": {
            "primary_score": 0.42,
            "secondary_score": 0.55,
            "per_mlp": [],
            "breakdowns": {
                "sampling": {
                    "flop_budget": 100,
                    "flops_used": 60,
                    "flops_remaining": 40,
                    "tracked_time_s": 0.015,
                    "untracked_time_s": 0.005,
                    "by_namespace": {
                        "sampling.sample_layer_statistics": {
                            "flops_used": 40,
                            "tracked_time_s": 0.01,
                        },
                        "sampling.draw_weights": {
                            "flops_used": 20,
                            "tracked_time_s": 0.005,
                        },
                    },
                },
                "estimator": {
                    "flop_budget": 100,
                    "flops_used": 30,
                    "flops_remaining": 70,
                    "tracked_time_s": 0.01,
                    "untracked_time_s": 0.002,
                    "by_namespace": {
                        "estimator.phase": {
                            "flops_used": 18,
                            "tracked_time_s": 0.006,
                        },
                        "estimator.estimator-client": {
                            "flops_used": 12,
                            "tracked_time_s": 0.004,
                        },
                    },
                },
            },
        },
    }


@contextmanager
def _noop_progress(*_args: Any, **_kwargs: Any):
    yield lambda _event: None


def test_smoke_test_falls_back_to_plain_text_when_rich_render_fails(monkeypatch, capsys) -> None:
    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress=None
    ) -> dict:
        return _sample_report()

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("rich boom")

    monkeypatch.setattr(
        cli,
        "render_human_report",
        fail_render,
        raising=False,
    )

    exit_code = cli.main(["smoke-test"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Rich dashboard unavailable (rich boom)" in captured.err
    assert "WhestBench Report (Plain Text)" in captured.out
    assert "Primary Score: 0.42" in captured.out


def test_smoke_test_plain_output_includes_next_steps_and_json_tip(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "run_default_report",
        lambda **_kwargs: _sample_report(),
    )

    exit_code = cli.main(["smoke-test", "--no-rich"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Next Steps" in captured.out
    assert "Create starter files you can edit." in captured.out
    assert "whest init ./my-estimator" in captured.out
    assert "Use --json for JSON output when calling from automated agents or UIs." in captured.out
    assert "Use --show-diagnostic-plots to include diagnostic plot panes." in captured.out


def test_participant_run_falls_back_to_plain_text_when_rich_render_fails(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )
    monkeypatch.setattr(cli, "_run_estimator_with_runner", lambda *_a, **_k: _sample_report())
    monkeypatch.setattr(
        cli,
        "_print_human_startup",
        lambda *_a, **_k: None,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "_progress_callback",
        _noop_progress,
        raising=False,
    )

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(cli, "render_human_results", fail_render, raising=False)

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "inprocess"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Rich dashboard unavailable (render failed)" in captured.err
    assert "WhestBench Report (Plain Text)" in captured.out


def test_render_plain_text_report_includes_breakdown_sections_and_summary() -> None:
    rendered = cli._render_plain_text_report(_sample_report())

    assert "Sampling Budget Breakdown:" in rendered
    assert "Estimator Budget Breakdown:" in rendered
    assert rendered.index("Sampling Budget Breakdown:") < rendered.index(
        "Estimator Budget Breakdown:"
    )
    assert "Total FLOPs: 60" in rendered
    assert "Tracked Time: 0.015000s" in rendered
    assert "Untracked Time: 0.005000s" in rendered
    assert "sampling.sample_layer_statistics" in rendered
    assert "sampling.draw_weights" in rendered
    assert "estimator.phase" in rendered
    assert "estimator.estimator-client" in rendered


def test_render_plain_text_report_includes_run_context_and_hardware_metadata() -> None:
    report = _sample_report()
    report["run_meta"]["host"] = {
        "hostname": "example-host",
        "os": "Darwin",
        "os_release": "25.3.0",
        "platform": "macOS-15-arm64",
        "machine": "arm64",
        "cpu_brand": "Apple M4",
        "cpu_count_logical": 10,
        "cpu_count_physical": 8,
        "ram_total_bytes": 17179869184,
        "python_version": "3.13.7",
        "numpy_version": "2.2.6",
    }
    report["run_meta"]["run_started_at_utc"] = "2026-03-01T00:00:00+00:00"
    report["run_meta"]["run_finished_at_utc"] = "2026-03-01T00:00:01+00:00"
    report["run_config"]["estimator_class"] = "CombinedEstimator"
    report["run_config"]["estimator_path"] = "examples/estimators/combined_estimator.py"

    rendered = cli._render_plain_text_report(report)

    for text in (
        "Estimator Class: CombinedEstimator",
        "Estimator Path: examples/estimators/combined_estimator.py",
        "Hardware & Runtime",
        "Host: example-host",
        "OS: Darwin",
        "CPU: Apple M4",
        "Python: 3.13.7",
    ):
        assert text in rendered


def test_error_code_mapping_for_validation_messages() -> None:
    bad_shape = "Predictions must have shape (2, 4), got (1, 4)."
    assert cli._error_code(ValueError(bad_shape), bad_shape) == "ESTIMATOR_BAD_SHAPE"
    non_finite = "Predictions must contain only finite values."
    assert cli._error_code(ValueError(non_finite), non_finite) == "ESTIMATOR_NON_FINITE"


def test_error_payload_shape_is_stable() -> None:
    payload = cli._error_payload(ValueError("bad row"), include_traceback=False)
    assert payload["ok"] is False
    assert payload["error"]["stage"] == "scoring"
    assert payload["error"]["code"] == "SCORING_VALIDATION_ERROR"
    assert payload["error"]["message"] == "bad row"
