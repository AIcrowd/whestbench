from __future__ import annotations

import json

from circuit_estimation.reporting import render_agent_report, render_human_report


def _sample_report(*, include_profile: bool = False) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": "1.0",
        "mode": "agent",
        "detail": "raw",
        "run_meta": {
            "run_started_at_utc": "2026-03-01T00:00:00+00:00",
            "run_finished_at_utc": "2026-03-01T00:00:01+00:00",
            "run_duration_s": 1.0,
        },
        "run_config": {
            "n_circuits": 2,
            "n_samples": 100,
            "width": 4,
            "max_depth": 3,
            "layer_count": 3,
            "budgets": [10, 100],
            "time_tolerance": 0.1,
            "profile_enabled": include_profile,
        },
        "circuits": [
            {"circuit_index": 0, "wire_count": 4, "layer_count": 3},
            {"circuit_index": 1, "wire_count": 4, "layer_count": 3},
        ],
        "results": {
            "final_score": 0.123,
            "score_direction": "lower_is_better",
            "by_budget_raw": [
                {
                    "budget": 10,
                    "score": 0.2,
                    "mse_by_layer": [0.1, 0.2, 0.3],
                    "time_ratio_by_layer": [1.0, 1.1, 1.2],
                    "adjusted_mse_by_layer": [0.1, 0.22, 0.36],
                    "timeout_flag_by_layer": [0.0, 0.0, 0.0],
                    "time_floor_flag_by_layer": [0.0, 0.0, 0.0],
                    "baseline_time_s_by_layer": [0.01, 0.01, 0.01],
                    "effective_time_s_by_layer": [0.01, 0.011, 0.012],
                },
                {
                    "budget": 100,
                    "score": 0.046,
                    "mse_by_layer": [0.05, 0.04, 0.03],
                    "time_ratio_by_layer": [1.0, 0.9, 0.8],
                    "adjusted_mse_by_layer": [0.05, 0.036, 0.024],
                    "timeout_flag_by_layer": [0.0, 0.0, 0.0],
                    "time_floor_flag_by_layer": [0.0, 0.0, 0.0],
                    "baseline_time_s_by_layer": [0.02, 0.02, 0.02],
                    "effective_time_s_by_layer": [0.02, 0.018, 0.016],
                },
            ],
        },
        "notes": [],
    }
    if include_profile:
        report["profile_calls"] = [
            {
                "budget": 10,
                "circuit_index": 0,
                "wire_count": 4,
                "layer_count": 3,
                "wall_time_s": 0.05,
                "cpu_time_s": 0.04,
                "rss_bytes": 12_345_678,
                "peak_rss_bytes": 15_000_000,
            }
        ]
    return report


def test_render_agent_mode_returns_pretty_json_only() -> None:
    report = _sample_report()
    rendered = render_agent_report(report)

    # Agent mode contract: machine-parseable, pretty JSON, no narrative framing.
    loaded = json.loads(rendered)
    assert loaded == report
    assert rendered.startswith("{\n")
    assert rendered.endswith("\n")


def test_render_human_mode_includes_expected_sections_without_profile() -> None:
    rendered = render_human_report(_sample_report(include_profile=False))

    # Human mode contract: high-level run summary plus budget and layer diagnostics.
    assert "Circuit Estimation Report" in rendered
    assert "Run Context" in rendered
    assert "Score Summary" in rendered
    assert "Budget Breakdown" in rendered
    assert "Layer Diagnostics" in rendered
    assert "Profiling" not in rendered


def test_render_human_mode_includes_profile_section_when_available() -> None:
    rendered = render_human_report(_sample_report(include_profile=True))

    assert "Profiling" in rendered
    assert "wall_time_s" in rendered
    assert "cpu_time_s" in rendered
    assert "rss_bytes" in rendered
    assert "peak_rss_bytes" in rendered
