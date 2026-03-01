from __future__ import annotations

from typing import Any

import pytest

from circuit_estimation.textual_dashboard.state import build_dashboard_state


def _sample_report(*, include_profile: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "run_meta": {
            "run_started_at_utc": "2026-03-01T00:00:00+00:00",
            "run_finished_at_utc": "2026-03-01T00:00:01+00:00",
            "run_duration_s": 1.0,
            "host": {
                "hostname": "example-host",
                "os": "Darwin",
                "os_release": "25.3.0",
                "platform": "macOS-26.3-arm64-arm-64bit-Mach-O",
                "machine": "arm64",
                "python_version": "3.13.7",
            },
        },
        "run_config": {
            "n_circuits": 2,
            "n_samples": 100,
            "width": 4,
            "max_depth": 3,
            "layer_count": 3,
            "budgets": [10, 100],
            "time_tolerance": 0.1,
        },
        "results": {
            "final_score": 0.123,
            "by_budget_raw": [
                {
                    "budget": 10,
                    "mse_by_layer": [0.1, 0.2, 0.3],
                    "mse_mean": 0.2,
                    "adjusted_mse": 0.22,
                    "call_time_ratio_mean": 1.1,
                    "call_effective_time_s_mean": 0.011,
                },
                {
                    "budget": 100,
                    "mse_by_layer": [0.05, 0.04, 0.03],
                    "mse_mean": 0.04,
                    "adjusted_mse": 0.036,
                    "call_time_ratio_mean": 0.9,
                    "call_effective_time_s_mean": 0.018,
                },
            ],
        },
    }
    if include_profile:
        report["profile_calls"] = [
            {
                "budget": 10,
                "circuit_index": 0,
                "wall_time_s": 0.05,
                "cpu_time_s": 0.04,
                "rss_bytes": 12_345_678,
                "peak_rss_bytes": 15_000_000,
            }
        ]
    return report


def test_state_derives_budget_extrema() -> None:
    state = build_dashboard_state(_sample_report())

    assert state.derived.final_score == 0.123
    assert state.derived.best_budget_score == 0.036
    assert state.derived.worst_budget_score == 0.22
    assert state.derived.score_spread == 0.184


def test_state_profile_flag_tracks_profile_presence() -> None:
    state = build_dashboard_state(_sample_report(include_profile=False))

    assert state.derived.has_profile is False


def test_state_exposes_host_metadata_for_hardware_pane() -> None:
    state = build_dashboard_state(_sample_report())

    assert state.derived.host_hostname == "example-host"
    assert state.derived.host_os == "Darwin"
    assert state.derived.host_release == "25.3.0"
    assert state.derived.host_platform == "macOS-26.3-arm64-arm-64bit-Mach-O"
    assert state.derived.host_machine == "arm64"
    assert state.derived.host_python_version == "3.13.7"


def test_state_derives_layer_summary_stats() -> None:
    state = build_dashboard_state(_sample_report())

    assert state.derived.layer_count == 3
    assert state.derived.layer_mse_min == pytest.approx(0.075)
    assert state.derived.layer_mse_max == pytest.approx(0.165)
    assert state.derived.layer_mse_mean == pytest.approx(0.12)
    assert state.derived.layer_mse_p05 == pytest.approx(0.075)
    assert state.derived.layer_mse_p95 == pytest.approx(0.165)
