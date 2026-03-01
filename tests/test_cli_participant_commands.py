from __future__ import annotations

import json
import tarfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

import circuit_estimation.cli as cli


def _sample_report() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "mode": "agent",
        "detail": "raw",
        "run_meta": {
            "run_started_at_utc": "2026-03-01T00:00:00+00:00",
            "run_finished_at_utc": "2026-03-01T00:00:01+00:00",
            "run_duration_s": 1.0,
            "host": {},
        },
        "run_config": {
            "n_circuits": 1,
            "n_samples": 10,
            "width": 4,
            "max_depth": 3,
            "layer_count": 3,
            "budgets": [10],
            "time_tolerance": 0.1,
            "profile_enabled": False,
        },
        "circuits": [{"circuit_index": 0, "wire_count": 4, "layer_count": 3}],
        "results": {
            "final_score": 0.42,
            "score_direction": "lower_is_better",
            "by_budget_raw": [],
        },
        "notes": [],
    }


def test_validate_command_returns_json_only_in_agent_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "validate_submission_entrypoint",
        lambda *_args, **_kwargs: {"ok": True, "class_name": "Estimator", "output_shape": [1, 4]},
    )

    exit_code = cli.main(["validate", "--estimator", "estimator.py", "--agent-mode"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "ok": True,
        "class_name": "Estimator",
        "output_shape": [1, 4],
    }


def test_run_command_renders_human_report_in_non_agent_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "score_submission_report", lambda *_args, **_kwargs: _sample_report())
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: "human report\n",
    )
    monkeypatch.setattr(
        cli,
        "render_agent_report",
        lambda _report: pytest.fail("agent renderer should not be called"),
    )

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "inprocess"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == "human report\n"


def test_package_command_writes_manifest_with_entrypoint_and_hashes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    estimator = tmp_path / "estimator.py"
    estimator.write_text(
        dedent(
            """
            import numpy as np
            from circuit_estimation import BaseEstimator, Circuit

            class Estimator(BaseEstimator):
                def predict(self, circuit: Circuit, budget: int):
                    for _ in range(circuit.d):
                        yield np.zeros((circuit.n,), dtype=np.float32)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "submission.tar.gz"

    exit_code = cli.main(
        [
            "package",
            "--estimator",
            str(estimator),
            "--output",
            str(artifact),
            "--agent-mode",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["artifact_path"] == str(artifact.resolve())

    with tarfile.open(artifact, "r:gz") as archive:
        names = set(archive.getnames())
        manifest_member = archive.extractfile("manifest.json")
        assert manifest_member is not None
        manifest = json.loads(manifest_member.read().decode("utf-8"))

    assert "estimator.py" in names
    assert "manifest.json" in names
    assert manifest["entrypoint"]["class"] == "Estimator"


def test_init_and_run_help_text_reference_examples_estimators_path() -> None:
    parser = cli._build_participant_parser()
    help_text = parser.format_help()
    assert "examples/estimators" in help_text
