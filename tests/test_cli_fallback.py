from __future__ import annotations

from typing import Any

import circuit_estimation.cli as cli


def _sample_report() -> dict[str, Any]:
    return {
        "mode": "human",
        "results": {
            "final_score": 0.42,
            "by_budget_raw": [],
        },
    }


def test_human_mode_falls_back_to_static_when_textual_unsupported(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "score_estimator_report", lambda *_a, **_k: _sample_report())
    monkeypatch.setattr(cli, "_supports_textual_dashboard", lambda: False)
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: "Circuit Estimation Report\n",
    )

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Textual UI unavailable" in captured.err
    assert "Circuit Estimation Report" in captured.out


def test_human_mode_falls_back_to_static_when_textual_launch_fails(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "score_estimator_report", lambda *_a, **_k: _sample_report())
    monkeypatch.setattr(cli, "_supports_textual_dashboard", lambda: True)

    def fail_launch(_report: dict[str, Any]) -> None:
        raise RuntimeError("launch failed")

    monkeypatch.setattr(cli, "_launch_textual_dashboard", fail_launch)
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: "Circuit Estimation Report\n",
    )

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Textual UI unavailable" in captured.err
    assert "launch failed" in captured.err
    assert "Circuit Estimation Report" in captured.out
