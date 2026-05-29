"""Dispatch + output tests for `whest login` / `whest submit`."""

from __future__ import annotations

from typing import List

from rich.console import Console as _RichConsole

import whestbench.aicrowd_config as cfg
import whestbench.cli as cli


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


def _stub_submit_pipeline(monkeypatch, *, registered=True, status_after=None, watch_raises=False):
    """Stub the whole AIcrowdClient so submit() runs offline.

    Matches the real client API (create_submission takes challenge_slug; the
    create response is `data`-wrapped with submission_id)."""
    calls: dict = {"created": None}

    class _FakeClient:
        def __init__(self, *, api_key, **kw):
            self.api_key = api_key

        def verify_identity(self):
            return 4242

        def resolve_challenge(self, slug):
            return 99

        def check_registration(self, *, challenge_id, participant_id):
            return registered

        def get_upload_details(self, *, challenge_slug):
            return {"url": "https://s3.test/upload", "fields": {"key": "subs/${filename}"}}

        def upload_to_s3(self, *, upload, file_path):
            return "subs/submission.tar.gz"

        def create_submission(self, *, challenge_slug, s3_key, description):
            calls["created"] = {"challenge_slug": challenge_slug, "s3_key": s3_key}
            return {"data": {"submission_id": 7777, "created_at": "t"}}

        def get_submission_status(self, sid):
            if watch_raises:
                from whestbench.aicrowd_client import AIcrowdAPIError

                raise AIcrowdAPIError(status=404, message="no participant status endpoint")
            return status_after or {"id": sid, "grading_status": "graded", "score": 0.9}

    monkeypatch.setattr(cli, "AIcrowdClient", _FakeClient, raising=False)
    return calls


def test_submit_watch_poll_failure_is_graceful(monkeypatch, tmp_path):
    # A successful submit must NOT be turned into a failure by a status-poll
    # error (the submission is created + grades asynchronously).
    _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(monkeypatch, watch_raises=True)
    art = tmp_path / "submission.tar.gz"
    art.write_bytes(b"\x1f\x8b\x08\x00fake")
    rc = cli.main(["submit", str(art), "--watch"])
    assert rc == 0


def test_submit_file_runs_full_hop_a(monkeypatch, tmp_path):
    captured = _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    calls = _stub_submit_pipeline(monkeypatch)
    art = tmp_path / "submission.tar.gz"
    art.write_bytes(b"\x1f\x8b\x08\x00fake")
    rc = cli.main(["submit", str(art)])
    assert rc == 0
    assert calls["created"]["challenge_slug"] == "arc-white-box-estimation-challenge-2026"
    assert any("7777" in line for line in captured)


def test_submit_estimator_packages_first(monkeypatch, tmp_path):
    _spy_console_print(monkeypatch)
    monkeypatch.setattr(cfg, "resolve_api_key", lambda explicit: "K")
    _stub_submit_pipeline(monkeypatch)
    packaged = tmp_path / "submission-packaged.tar.gz"
    packaged.write_bytes(b"\x1f\x8b\x08\x00fake")
    monkeypatch.setattr(cli, "package_submission", lambda *a, **k: packaged)
    est = tmp_path / "estimator.py"
    est.write_text("from whestbench import BaseEstimator\nclass Estimator(BaseEstimator): ...\n")
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
    art = tmp_path / "submission.tar.gz"
    art.write_bytes(b"x")
    rc = cli.main(["submit", str(art)])
    assert rc != 0
