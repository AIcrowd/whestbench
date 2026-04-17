from __future__ import annotations

import json
import tarfile
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

import whestbench.cli as cli


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
        lambda _report, *, show_diagnostic_plots=False, debug=False: "human report\n",
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
    def fake_progress(total: int, n_mlps: int, gen_label: str = "Generating MLPs"):
        observed["total"] = total
        observed["gen_label"] = gen_label
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
        lambda _report, *, show_diagnostic_plots=False, debug=False: "human report\n",
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
            from whestbench import BaseEstimator
            from whestbench.domain import MLP

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


# --- `whest run --dataset` + `--n-mlps` integration --------------------------


def _write_fake_dataset(path: Path, n_mlps: int, width: int = 4, depth: int = 2) -> None:
    """Write a .npz file that `load_dataset` accepts."""
    import numpy as np

    rng = np.random.default_rng(0)
    weights = rng.standard_normal((n_mlps, depth, width, width)).astype(np.float32)
    all_layer_means = rng.standard_normal((n_mlps, depth, width)).astype(np.float32)
    final_means = rng.standard_normal((n_mlps, width)).astype(np.float32)
    avg_variances = np.ones(n_mlps, dtype=np.float64)
    metadata = {
        "schema_version": "2.1",
        "created_at_utc": "2026-04-17T00:00:00+00:00",
        "seed": 0,
        "n_mlps": n_mlps,
        "n_samples": 100,
        "width": width,
        "depth": depth,
        "flop_budget": 1_000_000,
        "hardware": {},
    }
    np.savez(
        path,
        metadata=np.array(json.dumps(metadata)),
        weights=weights,
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=avg_variances,
    )


def _patch_run_command_happy_path(monkeypatch: pytest.MonkeyPatch, captured_kwargs: dict) -> None:
    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )
    monkeypatch.setattr(cli, "rich_tqdm", None, raising=False)
    monkeypatch.setattr(cli, "_print_human_startup", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(cli, "_progress_callback", _noop_progress, raising=False)
    monkeypatch.setattr(
        cli,
        "render_human_results",
        lambda _report, *, show_diagnostic_plots=False, debug=False: "human report\n",
        raising=False,
    )

    def fake_run_estimator_with_runner(*_args: Any, **kwargs: Any) -> dict:
        captured_kwargs.update(kwargs)
        return _sample_report()

    monkeypatch.setattr(cli, "_run_estimator_with_runner", fake_run_estimator_with_runner)


def test_run_with_dataset_uses_bundle_mlps_and_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ds_path = tmp_path / "ds.npz"
    _write_fake_dataset(ds_path, n_mlps=4)

    observed: dict = {}
    _patch_run_command_happy_path(monkeypatch, observed)

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            "estimator.py",
            "--runner",
            "inprocess",
            "--dataset",
            str(ds_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    # No --n-mlps passed: should fall back to the dataset's full count.
    assert observed["n_mlps"] == 4
    assert observed["contest_data"] is not None
    # ContestData carries 4 MLPs, all from the bundle (no regeneration).
    assert len(observed["contest_data"].mlps) == 4
    # sampling_budget_breakdown is None — ground truth was precomputed.
    assert observed["contest_data"].sampling_budget_breakdown is None


def test_run_with_dataset_honors_smaller_n_mlps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ds_path = tmp_path / "ds.npz"
    _write_fake_dataset(ds_path, n_mlps=5)

    observed: dict = {}
    _patch_run_command_happy_path(monkeypatch, observed)

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            "estimator.py",
            "--runner",
            "inprocess",
            "--dataset",
            str(ds_path),
            "--n-mlps",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert "Warning" not in captured.err
    assert observed["n_mlps"] == 2
    assert len(observed["contest_data"].mlps) == 2
    assert observed["contest_data"].spec.n_mlps == 2


def test_run_with_dataset_clamps_and_warns_when_n_mlps_exceeds_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ds_path = tmp_path / "ds.npz"
    _write_fake_dataset(ds_path, n_mlps=3)

    observed: dict = {}
    _patch_run_command_happy_path(monkeypatch, observed)

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            "estimator.py",
            "--runner",
            "inprocess",
            "--dataset",
            str(ds_path),
            "--n-mlps",
            "10",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert "exceeds dataset size" in captured.err
    assert "using 3" in captured.err
    assert observed["n_mlps"] == 3
    assert len(observed["contest_data"].mlps) == 3


def test_run_without_dataset_defaults_n_mlps_to_ten(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict = {}
    _patch_run_command_happy_path(monkeypatch, observed)

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            "estimator.py",
            "--runner",
            "inprocess",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert observed["n_mlps"] == 10
    # Without a dataset, no contest_data is passed — runner builds it.
    assert observed["contest_data"] is None
    assert observed["contest_spec"].n_mlps == 10


def test_run_without_dataset_honors_explicit_n_mlps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict = {}
    _patch_run_command_happy_path(monkeypatch, observed)

    exit_code = cli.main(
        [
            "run",
            "--estimator",
            "estimator.py",
            "--runner",
            "inprocess",
            "--n-mlps",
            "7",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert observed["n_mlps"] == 7
    assert observed["contest_data"] is None
    assert observed["contest_spec"].n_mlps == 7


def test_run_seed_includes_seed_in_run_config_and_replays_inputs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict = {}

    def fake_run_estimator_with_runner(*_args: Any, **kwargs: Any) -> dict:
        contest_spec = kwargs["contest_spec"]
        observed.setdefault("seeds", []).append(contest_spec.seed)
        observed.setdefault("specs", []).append(
            (
                contest_spec.width,
                contest_spec.depth,
                contest_spec.n_mlps,
                contest_spec.flop_budget,
                contest_spec.ground_truth_samples,
                contest_spec.seed,
            )
        )
        payload = _sample_report()
        payload["run_config"]["seed"] = contest_spec.seed
        return payload

    monkeypatch.setattr(cli, "_run_estimator_with_runner", fake_run_estimator_with_runner)

    first = cli.main(
        [
            "run",
            "--estimator",
            "estimator.py",
            "--runner",
            "inprocess",
            "--seed",
            "2026",
            "--json",
        ]
    )
    first_out = capsys.readouterr().out
    first_payload = json.loads(first_out)

    second = cli.main(
        [
            "run",
            "--estimator",
            "estimator.py",
            "--runner",
            "inprocess",
            "--seed",
            "2026",
            "--json",
        ]
    )
    second_out = capsys.readouterr().out
    second_payload = json.loads(second_out)

    assert first == 0
    assert second == 0
    assert first_payload["run_config"]["seed"] == 2026
    assert second_payload["run_config"]["seed"] == 2026
    assert observed["seeds"] == [2026, 2026]
    assert observed["specs"][0] == observed["specs"][1]


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
        lambda _report, *, show_diagnostic_plots=False, debug=False: "human report\n",
    )
    monkeypatch.setattr(cli.sys, "argv", ["whest", "smoke-test"])

    exit_code = cli.main(None)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "human report" in captured.out
    assert "Next Steps" in captured.out


# --- `whest run` error propagation ------------------------------------------


def _write_broken_predict_estimator(path: Path, message: str = "intentional predict crash") -> None:
    path.write_text(
        dedent(
            f"""
            from whestbench import BaseEstimator

            class Estimator(BaseEstimator):
                def predict(self, mlp, budget):
                    raise RuntimeError({message!r})
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _write_tiny_dataset(path: Path, n_mlps: int = 2, width: int = 4, depth: int = 2) -> None:
    """Write a minimal valid `.npz` dataset so `whest run --dataset` bypasses
    the real `make_contest` (which uses a 100x16 contest spec by default).

    This is the single biggest reason these tests run in milliseconds rather
    than seconds: `make_contest_from_bundle` just slices arrays, no compute.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    weights = rng.standard_normal((n_mlps, depth, width, width)).astype(np.float32)
    all_layer_means = rng.standard_normal((n_mlps, depth, width)).astype(np.float32)
    final_means = rng.standard_normal((n_mlps, width)).astype(np.float32)
    avg_variances = np.ones(n_mlps, dtype=np.float64)
    metadata = {
        "schema_version": "2.1",
        "created_at_utc": "2026-04-17T00:00:00+00:00",
        "seed": 0,
        "n_mlps": n_mlps,
        "n_samples": 10,
        "width": width,
        "depth": depth,
        "flop_budget": 1_000_000,
        "hardware": {},
    }
    np.savez(
        path,
        metadata=np.array(json.dumps(metadata)),
        weights=weights,
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=avg_variances,
    )


def _tiny_run_argv(estimator_path: Path, dataset_path: Path) -> list[str]:
    """Invoke `whest run` against a tiny dataset — no real sampling happens."""
    return [
        "run",
        "--estimator",
        str(estimator_path),
        "--runner",
        "local",
        "--dataset",
        str(dataset_path),
        "--no-rich",
    ]


def test_run_exits_1_when_predict_raises_local_runner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    estimator = tmp_path / "broken.py"
    _write_broken_predict_estimator(estimator)
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset)

    exit_code = cli.main(_tiny_run_argv(estimator, dataset))
    captured = capsys.readouterr()

    assert exit_code == 1
    # --no-rich uses plain-text output (no Rich panel), but the per-MLP error
    # text is still rendered and the stderr summary must be present.
    assert "intentional predict crash" in captured.out
    assert "raised during predict" in captured.err


def test_run_debug_includes_traceback_in_human_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    estimator = tmp_path / "broken.py"
    _write_broken_predict_estimator(estimator)
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset)

    exit_code = cli.main(_tiny_run_argv(estimator, dataset) + ["--debug"])
    captured = capsys.readouterr()

    assert exit_code == 1
    combined = captured.out + captured.err
    assert "Traceback" in combined
    assert "RuntimeError" in combined


def test_run_fail_fast_propagates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    estimator = tmp_path / "broken.py"
    _write_broken_predict_estimator(estimator)
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset)

    exit_code = cli.main(_tiny_run_argv(estimator, dataset) + ["--fail-fast"])
    captured = capsys.readouterr()

    assert exit_code == 1
    # Top-level handler prints `Error [predict:PREDICT_ERROR]: ...`.
    combined = captured.out + captured.err
    assert "PREDICT_ERROR" in combined


def test_run_json_output_includes_traceback_per_mlp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    estimator = tmp_path / "broken.py"
    _write_broken_predict_estimator(estimator)
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset, n_mlps=2)

    exit_code = cli.main(_tiny_run_argv(estimator, dataset) + ["--json"])
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.out)
    per_mlp = payload["results"]["per_mlp"]
    assert len(per_mlp) == 2
    for entry in per_mlp:
        assert entry["error_code"] == "PREDICT_ERROR"
        assert "intentional predict crash" in entry["error"]
        assert isinstance(entry["traceback"], str)
        assert "RuntimeError" in entry["traceback"]


def test_run_json_output_includes_validation_details_for_shape_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    estimator = tmp_path / "wrong_shape.py"
    estimator.write_text(
        dedent(
            """
            import whest as we
            from whestbench import BaseEstimator

            class Estimator(BaseEstimator):
                def predict(self, mlp, budget):
                    return we.ones((mlp.width, mlp.depth), dtype=we.float32)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset, n_mlps=1)

    exit_code = cli.main(_tiny_run_argv(estimator, dataset) + ["--json", "--runner", "subprocess"])
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.out)
    per_mlp = payload["results"]["per_mlp"]
    assert len(per_mlp) == 1
    entry = per_mlp[0]
    assert entry["error_code"] == "PREDICT_ERROR"
    assert isinstance(entry["error"], dict)
    assert "message" in entry["error"]
    assert "details" in entry["error"]
    assert entry["error"]["details"]["expected_shape"] == [2, 4]
    assert entry["error"]["details"]["got_shape"] == [4, 2]


def test_run_plain_output_shows_validation_hint_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    estimator = tmp_path / "wrong_shape.py"
    estimator.write_text(
        dedent(
            """
            import whest as we
            from whestbench import BaseEstimator

            class Estimator(BaseEstimator):
                def predict(self, mlp, budget):
                    return we.ones((mlp.width, mlp.depth), dtype=we.float32)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset, n_mlps=1)

    exit_code = cli.main(_tiny_run_argv(estimator, dataset))
    captured = capsys.readouterr()

    assert exit_code == 1
    out = captured.out
    assert "Estimator Errors" in out
    assert "MLP 0 [ValueError]:" in out
    assert "Predictions must have shape (2, 4), got (4, 2)." in out
    assert "Expected shape: [2, 4]" in out
    assert "Got shape: [4, 2]" in out
    assert "Hint:" in out
    assert "'message':" not in out


def test_run_budget_exhausted_does_not_set_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Budget exhaustion is a legitimate scoring outcome, not an error.

    Uses a trivially tiny budget and a predict that raises
    BudgetExhaustedError directly (no real FLOP loop), so the test stays
    fast.
    """
    estimator = tmp_path / "hungry.py"
    estimator.write_text(
        dedent(
            """
            import whest as we
            from whestbench import BaseEstimator

            class Estimator(BaseEstimator):
                def predict(self, mlp, budget):
                    raise we.BudgetExhaustedError('test', flop_cost=0, flops_remaining=0)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset)

    exit_code = cli.main(_tiny_run_argv(estimator, dataset) + ["--json"])
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)
    per_mlp = payload["results"]["per_mlp"]
    assert all("error" not in entry for entry in per_mlp)
    assert all(entry["budget_exhausted"] for entry in per_mlp)
