"""Verify the unified say.intent / say.ok bookends fire for non-dataset verbs.

Phase 7 of the UX overhaul sweeps the remaining CLI commands so they all
emit a `say.intent` header and a `say.ok` footer for visible operations.
These tests stub out underlying work and confirm both lines appear in the
captured Rich-console output for each verb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest
from rich.console import Console as _RichConsole

import whestbench.cli as cli


def _spy_console_print(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Capture every ``Console.print`` first-arg string for inspection.

    Real print still fires so test failures show captured output in
    pytest's stdout/stderr.
    """
    captured: List[str] = []
    original_print = _RichConsole.print

    def spy_print(self: _RichConsole, *args: Any, **kwargs: Any) -> Any:
        if args:
            captured.append(str(args[0]))
        return original_print(self, *args, **kwargs)

    monkeypatch.setattr(_RichConsole, "print", spy_print)
    return captured


def test_doctor_emits_intent_and_ok_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`whest doctor` brackets the probe sweep with say.intent / say.ok."""
    captured = _spy_console_print(monkeypatch)

    # Stub the actual probe sweep so we test the wrapping, not the checks.
    monkeypatch.setattr(
        "whestbench.doctor.run_all",
        lambda *_a, **_k: [
            {
                "name": "python_version",
                "label": "Python version",
                "status": "ok",
                "detail": "3.12.0",
                "fix_hint": None,
            }
        ],
    )

    rc = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert rc == 0
    joined = "\n".join(captured)
    assert "Running whestbench doctor" in joined
    # Footer is wired via the unified say.ok helper.
    assert "Ran 1 check" in joined
    assert "✓" in (joined + out)


def test_doctor_json_suppresses_say_bookends(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """In `--json` mode, doctor must not leak intent/ok lines into stdout."""
    monkeypatch.setattr(
        "whestbench.doctor.run_all",
        lambda *_a, **_k: [
            {
                "name": "python_version",
                "label": "Python version",
                "status": "ok",
                "detail": "3.12.0",
                "fix_hint": None,
            }
        ],
    )

    rc = cli.main(["doctor", "--json"])
    captured = capsys.readouterr()

    assert rc == 0
    # The JSON payload must remain pure JSON on stdout — intent/ok bookends
    # are suppressed via quiet=json_output.
    assert "Running whestbench doctor" not in captured.out
    assert "✓" not in captured.out
    assert captured.out.lstrip().startswith("{")


def test_package_emits_intent_and_ok_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`whest package` brackets the archive write with say.intent / say.ok."""
    captured = _spy_console_print(monkeypatch)

    estimator = tmp_path / "estimator.py"
    estimator.write_text("class Estimator:\n    pass\n")
    output = tmp_path / "out.tar.gz"

    def fake_package(*_args: Any, **kwargs: Any) -> Path:
        # Honor the progress callback (so the bar fires) and write a small
        # tarball-ish blob to the expected output path.
        cb = kwargs.get("progress")
        if cb is not None:
            cb(64)
        path = Path(kwargs["output_path"])
        path.write_bytes(b"\0" * 100)
        return path

    monkeypatch.setattr(cli, "package_submission", fake_package)

    rc = cli.main(
        [
            "package",
            "--estimator",
            str(estimator),
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    joined = "\n".join(captured)
    assert "Packaging" in joined
    assert str(estimator) in joined
    # say.ok with size + duration suffix.
    assert "Wrote" in joined
    assert "✓" in joined


def test_validate_emits_intent_and_ok_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`whest validate` brackets the checks with say.intent / say.ok."""
    captured = _spy_console_print(monkeypatch)

    monkeypatch.setattr(
        cli,
        "_run_validate_checks",
        lambda *_a, **_k: {
            "ok": True,
            "class_name": "Estimator",
            "module_name": "_submission",
            "output_shape": [2, 4],
            "checks": [
                {"name": "class resolved", "status": "ok", "detail": "Estimator"},
            ],
        },
    )

    rc = cli.main(["validate", "--estimator", "estimator.py"])
    capsys.readouterr()  # drain so pytest's captured output stays predictable

    assert rc == 0
    joined = "\n".join(captured)
    assert "Validating estimator estimator.py" in joined
    assert "Validation passed" in joined
    assert "✓" in joined


def test_validate_json_suppresses_say_bookends(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """In `--json` mode, validate must keep stdout clean for downstream tools."""
    monkeypatch.setattr(
        cli,
        "validate_submission_entrypoint",
        lambda *_a, **_k: {"ok": True, "class_name": "Estimator", "output_shape": [2, 4]},
    )

    rc = cli.main(["validate", "--estimator", "estimator.py", "--json"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Validating estimator" not in captured.out
    assert "✓" not in captured.out
    assert captured.out.lstrip().startswith("{")


def test_init_emits_intent_and_ok_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """`whest init` brackets template writes with say.intent / say.ok."""
    captured = _spy_console_print(monkeypatch)

    target = tmp_path / "demo"
    monkeypatch.setattr(
        cli,
        "_write_init_template",
        lambda _path: [str(target / "estimator.py"), str(target / "requirements.txt")],
    )

    rc = cli.main(["init", str(target)])
    capsys.readouterr()

    assert rc == 0
    joined = "\n".join(captured)
    assert "Initializing starter estimator" in joined
    assert "Created 2 files" in joined
    assert "✓" in joined


def test_init_emits_ok_when_already_present(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """When `_write_init_template` reports nothing new, say.ok still fires."""
    captured = _spy_console_print(monkeypatch)

    monkeypatch.setattr(cli, "_write_init_template", lambda _path: [])

    rc = cli.main(["init", str(tmp_path / "demo")])
    capsys.readouterr()

    assert rc == 0
    joined = "\n".join(captured)
    assert "Initializing starter estimator" in joined
    # No-op branch surfaces a softer footer but still uses ✓.
    assert "Starter files already present" in joined
    assert "✓" in joined


def test_profile_simulation_emits_intent_and_ok_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`whest profile-simulation` brackets the sweep with say.intent / say.ok."""
    captured = _spy_console_print(monkeypatch)

    import whestbench.profiler as _profiler

    monkeypatch.setattr(
        _profiler,
        "run_profile",
        lambda **_kwargs: ("Simulation Profile output\n", {"hardware": {}, "timing": []}),
        raising=False,
    )

    rc = cli.main(["profile-simulation", "--preset", "super-quick"])
    capsys.readouterr()

    assert rc == 0
    joined = "\n".join(captured)
    assert "Profiling simulation backends" in joined
    assert "preset=super-quick" in joined
    assert "Profile completed" in joined
    assert "✓" in joined


def test_smoke_test_rich_emits_intent_and_ok_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`whest smoke-test --format rich` brackets the run with say.intent / say.ok."""
    captured = _spy_console_print(monkeypatch)

    # Stub the heavyweight run with a minimal smoke report so the rich
    # presenter has something coherent to render. The shape comes from the
    # presentation builder's expected keys.
    report = {
        "schema_version": "1.1",
        "mode": "human",
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
            "flop_budget": 40000,
            "profile_enabled": False,
        },
        "results": {
            "primary_score": 0.42,
            "secondary_score": 0.55,
            "per_mlp": [],
        },
        "notes": [],
    }
    monkeypatch.setattr(cli, "run_default_report", lambda **_k: report)

    rc = cli.main(["smoke-test", "--format", "rich"])
    capsys.readouterr()

    assert rc == 0
    joined = "\n".join(captured)
    assert "Running smoke test against CombinedEstimator" in joined
    assert "Smoke test completed" in joined
    assert "✓" in joined
