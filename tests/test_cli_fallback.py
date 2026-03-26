from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import network_estimation.cli as cli


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
            "estimator_budget": 40000,
        },
        "results": {
            "primary_score": 0.42,
            "secondary_score": 0.55,
            "per_mlp": [],
        },
    }


@contextmanager
def _noop_progress(*_args: Any, **_kwargs: Any):
    yield lambda _event: None


def test_smoke_test_falls_back_to_plain_text_when_rich_render_fails(monkeypatch, capsys) -> None:
    def fake_run_default_report(*, profile: bool = False, detail: str = "raw", progress=None) -> dict:
        return _sample_report()

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("rich boom")

    monkeypatch.setattr(
        cli,
        "render_human_report",
        fail_render,
    )

    exit_code = cli.main(["smoke-test"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Rich dashboard unavailable (rich boom)" in captured.err
    assert "Network Estimation Report (Plain Text)" in captured.out
    assert "Primary Score: 0.42" in captured.out


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
    assert "Network Estimation Report (Plain Text)" in captured.out


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
