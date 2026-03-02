from __future__ import annotations

from typing import Any

import circuit_estimation.cli as cli


def _sample_report() -> dict[str, Any]:
    return {
        "mode": "human",
        "run_meta": {
            "run_duration_s": 1.25,
        },
        "run_config": {
            "n_circuits": 2,
            "n_samples": 50,
            "width": 4,
            "max_depth": 3,
            "budgets": [10, 100],
        },
        "results": {
            "final_score": 0.42,
            "by_budget_raw": [
                {"budget": 10, "score": 0.6},
                {"budget": 100, "score": 0.3},
            ],
        },
    }


def test_human_mode_falls_back_to_plain_text_when_rich_render_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "score_estimator_report", lambda *_a, **_k: _sample_report())

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("rich boom")

    monkeypatch.setattr(
        cli,
        "render_human_report",
        fail_render,
    )

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Rich dashboard unavailable (rich boom)" in captured.err
    assert "Circuit Estimation Report (Plain Text)" in captured.out
    assert "Final Score: 0.42" in captured.out
    assert "Best Budget Score: 0.3" in captured.out


def test_participant_run_falls_back_to_plain_text_when_rich_render_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "score_submission_report", lambda *_a, **_k: _sample_report())

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(
        cli,
        "render_human_report",
        fail_render,
    )

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "inprocess"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Rich dashboard unavailable (render failed)" in captured.err
    assert "Circuit Estimation Report (Plain Text)" in captured.out


def test_error_code_mapping_for_stream_contract_messages() -> None:
    not_iterable = "Estimator must return an iterator of depth-row outputs."
    assert (
        cli._error_code(ValueError(not_iterable), not_iterable) == "ESTIMATOR_STREAM_NOT_ITERABLE"
    )
    too_many = "Estimator emitted more than max_depth rows."
    assert cli._error_code(ValueError(too_many), too_many) == "ESTIMATOR_STREAM_TOO_MANY_ROWS"
    too_few = "Estimator must emit exactly max_depth rows."
    assert cli._error_code(ValueError(too_few), too_few) == "ESTIMATOR_STREAM_TOO_FEW_ROWS"
    bad_shape = "Estimator row at depth 0 must have shape (4,), got (1,)."
    assert cli._error_code(ValueError(bad_shape), bad_shape) == "ESTIMATOR_STREAM_BAD_ROW_SHAPE"
    non_finite = "Estimator row at depth 1 must contain finite values."
    assert cli._error_code(ValueError(non_finite), non_finite) == "ESTIMATOR_STREAM_NON_FINITE_ROW"


def test_error_payload_shape_is_stable() -> None:
    payload = cli._error_payload(ValueError("bad row"), include_traceback=False)
    assert payload["ok"] is False
    assert payload["error"]["stage"] == "scoring"
    assert payload["error"]["code"] == "SCORING_VALIDATION_ERROR"
    assert payload["error"]["message"] == "bad row"
