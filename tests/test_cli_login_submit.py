"""Dispatch + output tests for `whest login` / `whest submit`."""

from __future__ import annotations

from pathlib import Path
from typing import List

from rich.console import Console as _RichConsole

import whestbench.aicrowd_config as cfg
import whestbench.cli as cli
from whestbench.packaging import package_submission

_VALID_ESTIMATOR = (
    "from whestbench import BaseEstimator\n"
    "class Estimator(BaseEstimator):\n"
    "    def predict(self, mlp, budget):\n"
    "        import flopscope.numpy as fnp\n"
    "        return fnp.zeros((mlp.depth, mlp.width))\n"
)


def _valid_artifact(tmp_path: Path) -> Path:
    """Build a real, valid submission archive.

    `whest submit` validates the archive locally before uploading (rejecting
    directory manifest entries and drift — whestbench#107), so submit-pipeline
    tests must hand it a genuine archive, not fake bytes."""
    src = tmp_path / "_src"
    src.mkdir(exist_ok=True)
    (src / "estimator.py").write_text(_VALID_ESTIMATOR, encoding="utf-8")
    out = tmp_path / "submission.tar.gz"
    package_submission(src / "estimator.py", output_path=out)
    return out


def _spy_console_print(monkeypatch) -> List[str]:
    captured: List[str] = []
    original = _RichConsole.print

    def spy(self, *args, **kwargs):
        if args:
            captured.append(str(args[0]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_RichConsole, "print", spy)
    return captured


def test_login_with_api_key_flag_saves_and_verifies(monkeypatch, tmp_path):
    captured = _spy_console_print(monkeypatch)
    saved = {}
    monkeypatch.setattr(cfg, "save_api_key", lambda k: saved.setdefault("key", k) or tmp_path)
    # Stub identity verification so no network call happens.
    monkeypatch.setattr(cli, "_aicrowd_verify_identity", lambda key: {"id": 1, "username": "alice"})
    rc = cli.main(["login", "--api-key", "KEY-XYZ"])
    assert rc == 0
    assert saved["key"] == "KEY-XYZ"
    assert any("alice" in line for line in captured)


def test_login_rejects_invalid_key(monkeypatch):
    _spy_console_print(monkeypatch)

    def boom(key):
        from whestbench.aicrowd_client import AIcrowdAPIError

        raise AIcrowdAPIError(status=401, message="bad key")

    monkeypatch.setattr(cli, "_aicrowd_verify_identity", boom)
    monkeypatch.setattr(
        cfg, "save_api_key", lambda k: (_ for _ in ()).throw(AssertionError("must not save"))
    )
    rc = cli.main(["login", "--api-key", "BAD"])
    assert rc != 0


def _stub_submit_pipeline(
    monkeypatch,
    *,
    registered=True,
    status_after=None,
    watch_raises=False,
    transient_polls=0,
    transient_forever=False,
    create_raises=False,
):
    """Stub the whole AIcrowdClient so submit() runs offline.

    - watch_raises=True       -> get_submission_status raises a permanent 404
    - transient_forever=True  -> always raises AIcrowdTransientError (503)
    - transient_polls=N       -> raises AIcrowdTransientError N times, then returns
    """
    calls: dict = {"created": None}

    class _FakeClient:
        def __init__(self, *, api_key, **kw):
            self.api_key = api_key
            self._poll_n = 0

        def verify_identity(self):
            return 4242

        def check_eligibility(self, *, challenge_slug):
            # Mirrors GET /api/v1/challenges/<slug>/eligibility. The `registered`
            # knob is kept so existing callers read the same, but it now means
            # "may this participant submit", which is what the CLI actually asks.
            if registered:
                return {
                    "submissions_allowed": True,
                    "denied_reason": None,
                    "message": None,
                    "rules_accepted": True,
                    "participation_terms_accepted": True,
                }
            return {
                "submissions_allowed": False,
                "denied_reason": "rules_not_accepted",
                "message": (
                    "Please accept challenge terms before making submission here: "
                    "www.aicrowd.com/challenges/c/challenge_rules"
                ),
                "rules_accepted": False,
                "participation_terms_accepted": True,
                "rules_url": "https://www.aicrowd.com/challenges/c/challenge_rules",
            }

        def get_upload_details(self, *, challenge_slug):
            return {"url": "https://s3.test/upload", "fields": {"key": "subs/${filename}"}}

        def upload_to_s3(self, *, upload, file_path):
            return "subs/submission.tar.gz"

        def create_submission(self, *, challenge_slug, s3_key, description):
            if create_raises:
                from whestbench.aicrowd_client import AIcrowdTransientError

                raise AIcrowdTransientError(status=503, message="maintenance")
            calls["created"] = {"challenge_slug": challenge_slug, "s3_key": s3_key}
            return {"data": {"submission_id": 7777, "created_at": "t"}}

        def get_submission_status(self, sid):
            from whestbench.aicrowd_client import AIcrowdAPIError, AIcrowdTransientError

            if watch_raises:
                raise AIcrowdAPIError(status=404, message="no participant status endpoint")
            if transient_forever:
                raise AIcrowdTransientError(status=503, message="maintenance")
            if transient_polls and self._poll_n < transient_polls:
                self._poll_n += 1
                raise AIcrowdTransientError(status=503, message="maintenance")
            return status_after or {
                "id": sid,
                "grading_status_cd": "graded",
                "grading_message": "Graded successfully",
                "score": 0.9,
            }

    monkeypatch.setattr(cli, "AIcrowdClient", _FakeClient, raising=False)
    return calls


def test_submit_watch_poll_failure_is_graceful(monkeypatch, tmp_path):
    # A successful submit must NOT be turned into a failure by a status-poll
    # error (the submission is created + grades asynchronously).
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(monkeypatch, watch_raises=True)
    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art), "--watch"])
    assert rc == 0


def test_submit_watch_reaches_graded_and_prints_score(monkeypatch, tmp_path):
    # --watch polls until grading_status_cd hits a terminal state and reports
    # the score (mirrors Api::SubmissionSerializer: grading_status_cd + score).
    monkeypatch.setattr("time.sleep", lambda *_: None)
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(
        monkeypatch,
        status_after={"id": 7777, "grading_status_cd": "graded", "score": 0.0845},
    )
    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art), "--watch"])
    assert rc == 0
    assert any("0.0845" in line for line in captured)
    assert any("Graded" in line for line in captured)


def test_submit_watch_failed_grading_returns_nonzero(monkeypatch, tmp_path):
    # A terminal `failed` grade surfaces the message and a non-zero exit code.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(
        monkeypatch,
        status_after={
            "id": 7777,
            "grading_status_cd": "failed",
            "grading_message": "boom",
        },
    )
    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art), "--watch"])
    assert rc == 1
    assert any("boom" in line for line in captured)


def test_submit_file_runs_full_hop_a(monkeypatch, tmp_path):
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    calls = _stub_submit_pipeline(monkeypatch)
    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art)])
    assert rc == 0
    assert calls["created"]["challenge_slug"] == "arc-white-box-estimation-challenge-2026"
    assert any("7777" in line for line in captured)


def test_submit_estimator_packages_first(monkeypatch, tmp_path):
    _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(monkeypatch)
    packaged = _valid_artifact(tmp_path)
    monkeypatch.setattr(cli, "package_submission", lambda *a, **k: packaged)
    est = tmp_path / "estimator.py"
    est.write_text(
        "from whestbench import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        return None\n"
    )
    rc = cli.main(["submit", "--estimator", str(est)])
    assert rc == 0


def test_submit_not_logged_in_errors(monkeypatch, tmp_path):
    _spy_console_print(monkeypatch)

    def boom(explicit):
        raise cfg.NotLoggedIn("nope")

    monkeypatch.setattr(cfg, "resolve_api_key", boom)
    art = tmp_path / "submission.tar.gz"
    art.write_bytes(b"x")
    rc = cli.main(["submit", str(art)])
    assert rc != 0


def test_submit_unregistered_errors(monkeypatch, tmp_path):
    _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(monkeypatch, registered=False)
    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art)])
    assert rc != 0


def test_submit_relays_aicrowds_own_refusal_and_stops_before_upload(monkeypatch, tmp_path):
    """A refused pre-flight must print what AIcrowd said, verbatim, and upload nothing.

    whest used to print a sentence of its own ("You are not registered for
    '<slug>'."), which was often wrong and buried the real reason. Relay AIcrowd's
    sentence instead.
    """
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(monkeypatch, registered=False)
    art = _valid_artifact(tmp_path)

    rc = cli.main(["submit", str(art)])

    assert rc != 0
    text = "\n".join(captured)
    assert "Please accept challenge terms" in text
    # whest must not invent a cause of its own alongside AIcrowd's.
    assert "You are not registered" not in text
    # And it must never tell someone to re-login: the key was already validated.
    assert "whest login" not in text


def test_submit_does_not_blame_the_rules_when_the_round_is_closed(monkeypatch, tmp_path):
    """The reported bug, at the CLI boundary: rules are fine, the round is shut.

    Nothing in the output may suggest accepting anything — that advice sent
    participants to re-accept rules that were already current.
    """
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(monkeypatch)

    import whestbench.cli as _cli

    closed = {
        "submissions_allowed": False,
        "denied_reason": "challenge_completed",
        "message": "Phase 1 is completed — submissions are closed. Phase 2 opens on Nov 02, 2026 00:00.",
        "rules_accepted": True,
        "participation_terms_accepted": True,
        "rules_url": "https://www.aicrowd.com/challenges/c/challenge_rules",
    }
    original = _cli.AIcrowdClient

    class _Closed(original):  # type: ignore[misc, valid-type]
        def check_eligibility(self, *, challenge_slug):
            return closed

    monkeypatch.setattr(_cli, "AIcrowdClient", _Closed)
    art = _valid_artifact(tmp_path)

    rc = cli.main(["submit", str(art)])

    assert rc != 0
    text = "\n".join(captured)
    assert "submissions are closed" in text
    assert "Phase 2 opens on" in text
    assert "accept" not in text.lower()
    assert "challenge_rules" not in text


def test_submit_watch_absorbs_transient_then_grades(monkeypatch, tmp_path):
    # Transient 503s during polling are silent; the watcher still reaches graded
    # and never prints the old scary "Couldn't poll grading status" warning.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(
        monkeypatch,
        transient_polls=2,
        status_after={"id": 7777, "grading_status_cd": "graded", "score": 0.0845},
    )
    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art), "--watch"])
    assert rc == 0
    assert any("Graded" in line for line in captured)
    assert not any("Couldn't poll grading status" in line for line in captured)


def test_submit_watch_deadline_detaches_cleanly(monkeypatch, tmp_path):
    # If the platform stays unreachable, --watch-timeout detaches with a friendly
    # tracking link and exit 0 (never a scary error, never rc=1).
    monkeypatch.setattr("time.sleep", lambda *_: None)
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(monkeypatch, transient_forever=True)
    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art), "--watch", "--watch-timeout", "0"])
    assert rc == 0
    assert any("Still grading" in line for line in captured)
    assert not any("Couldn't poll grading status" in line for line in captured)


def test_submit_watch_permanent_error_degrades_gracefully(monkeypatch, tmp_path):
    # A permanent 404 (deployment without a status endpoint) degrades immediately
    # to a tracking hint, exit 0 — it must not spin until the deadline.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(monkeypatch, watch_raises=True)
    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art), "--watch", "--watch-timeout", "0"])
    assert rc == 0
    assert any("view it at" in line for line in captured)
    assert not any("Couldn't poll grading status" in line for line in captured)


def test_submit_path_transient_exhaustion_reports_failure(monkeypatch, tmp_path):
    # When a submit-path call exhausts its retries (transient error bubbles up),
    # the CLI surfaces a normal "Submission failed" error and a nonzero exit —
    # never a silent success.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(monkeypatch, create_raises=True)
    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art)])
    assert rc != 0
    assert any("HTTP 503" in line for line in captured)


def test_submit_failure_prints_message_and_hint(monkeypatch, tmp_path):
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")

    from whestbench.aicrowd_client import AIcrowdNotAllowedError

    _stub_submit_pipeline(monkeypatch)
    # After _stub_submit_pipeline, cli.AIcrowdClient IS _FakeClient — patch its method.
    monkeypatch.setattr(
        cli.AIcrowdClient,
        "create_submission",
        lambda self, **kw: (_ for _ in ()).throw(
            AIcrowdNotAllowedError(
                status=403, message="Submissions are not open.", op="creating your submission"
            )
        ),
    )

    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art)])
    assert rc != 0
    assert any("Submissions are not open." in line for line in captured)
    assert any("tip:" in line and "challenge page" in line for line in captured)


def test_submit_failure_json_has_error_code_and_status(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    from whestbench.aicrowd_client import AIcrowdNotAllowedError

    _stub_submit_pipeline(monkeypatch)
    # After _stub_submit_pipeline, cli.AIcrowdClient IS _FakeClient — patch its method.
    monkeypatch.setattr(
        cli.AIcrowdClient,
        "create_submission",
        lambda self, **kw: (_ for _ in ()).throw(
            AIcrowdNotAllowedError(
                status=403, message="Submissions are not open.", op="creating your submission"
            )
        ),
    )
    art = _valid_artifact(tmp_path)
    rc = cli.main(["submit", str(art), "--json"])
    assert rc != 0
    import json as _j

    payload = _j.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert payload["error_code"] == "not_allowed"
    assert payload["status"] == 403
    assert "Submissions are not open." in payload["error"]
