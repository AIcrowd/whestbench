from __future__ import annotations

import json
import tarfile
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

import network_estimation.cli as cli


def _sample_report() -> dict:
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
            "n_mlps": 1,
            "width": 4,
            "depth": 3,
            "estimator_budget": 40000,
            "profile_enabled": False,
        },
        "results": {
            "primary_score": 0.42,
            "secondary_score": 0.55,
            "per_mlp": [],
        },
        "notes": [],
    }


@contextmanager
def _noop_progress(*_args: Any, **_kwargs: Any):
    yield lambda _event: None


def test_validate_command_returns_json_only_with_json_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "validate_submission_entrypoint",
        lambda *_args, **_kwargs: {"ok": True, "class_name": "Estimator", "output_shape": [2, 4]},
    )

    exit_code = cli.main(["validate", "--estimator", "estimator.py", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "ok": True,
        "class_name": "Estimator",
        "output_shape": [2, 4],
    }


def test_run_command_renders_human_report_in_non_agent_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )
    monkeypatch.setattr(
        cli, "_run_estimator_with_runner", lambda *_args, **_kwargs: _sample_report()
    )
    monkeypatch.setattr(cli, "_print_human_startup", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(cli, "_progress_callback", _noop_progress, raising=False)
    monkeypatch.setattr(
        cli,
        "render_human_results",
        lambda _report, *, show_diagnostic_plots=False: "human report\n",
        raising=False,
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
    assert "human report\n" in captured.out


def test_run_command_human_mode_prints_startup_and_uses_progress_callback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict = {}

    @contextmanager
    def fake_progress(total: int, n_mlps: int):
        observed["total"] = total
        observed["progress_opened"] = True
        yield lambda _event: None
        observed["progress_closed"] = True

    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "MyEstimator"})(),
        raising=False,
    )
    monkeypatch.setattr(cli, "rich_tqdm", None, raising=False)
    monkeypatch.setattr(
        cli,
        "_print_human_startup",
        lambda _pre_report, *, estimator_class, estimator_path: observed.update(
            {
                "estimator_class": estimator_class,
                "estimator_path": str(estimator_path),
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(cli, "_progress_callback", fake_progress, raising=False)

    def fake_run_estimator_with_runner(*_args: Any, **kwargs: Any) -> dict:
        observed["scoring_progress_cb"] = kwargs.get("progress")
        return _sample_report()

    monkeypatch.setattr(
        cli,
        "_run_estimator_with_runner",
        fake_run_estimator_with_runner,
    )
    monkeypatch.setattr(
        cli,
        "render_human_results",
        lambda _report, *, show_diagnostic_plots=False: "human report\n",
        raising=False,
    )

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            "estimator.py",
            "--runner",
            "inprocess",
            "--n-mlps",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "human report\n" in captured.out
    assert observed["total"] == 2
    assert observed["progress_opened"] is True
    assert observed["progress_closed"] is True
    assert callable(observed["scoring_progress_cb"])


def test_run_command_json_mode_skips_human_startup_and_progress(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli, "_run_estimator_with_runner", lambda *_args, **_kwargs: _sample_report()
    )
    monkeypatch.setattr(
        cli,
        "_print_human_startup",
        lambda *_a, **_k: pytest.fail("human startup should not run in json mode"),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "_progress_callback",
        lambda *_a, **_k: pytest.fail("progress should not run in json mode"),
        raising=False,
    )

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "inprocess", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out)["mode"] == "agent"


def test_package_command_writes_manifest_with_entrypoint_and_hashes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    estimator = tmp_path / "estimator.py"
    estimator.write_text(
        dedent(
            """
            import numpy as np
            from network_estimation import BaseEstimator
            from network_estimation.domain import MLP

            class Estimator(BaseEstimator):
                def predict(self, mlp: MLP, budget: int):
                    return np.zeros((mlp.depth, mlp.width), dtype=np.float32)
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
            "--json",
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


def test_main_uses_sys_argv_when_argv_is_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress=None
    ) -> dict:
        return _sample_report()

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: "human report\n",
    )
    monkeypatch.setattr(cli.sys, "argv", ["nestim", "smoke-test"])

    exit_code = cli.main(None)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "human report" in captured.out
    assert "Next Steps" in captured.out
