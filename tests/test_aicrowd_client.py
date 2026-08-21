"""AIcrowd REST client — call-sequence tests via httpx MockTransport.

Pins the contract verified against the AIcrowd Rails source: Token auth, the
presigned-POST `data.{url,fields}` shape, and the NESTED submission-create body
(`submission.submission_files_attributes[].submission_file_s3_key`).
"""

from __future__ import annotations

import json as _json

import httpx
import pytest

import whestbench.aicrowd_client as client_mod
from whestbench.aicrowd_client import (
    _RETRYABLE_STATUS,
    POLL_RETRY,
    SUBMIT_RETRY,
    AIcrowdAPIError,
    AIcrowdAuthError,
    AIcrowdClient,
    AIcrowdNotAllowedError,
    AIcrowdNotFoundError,
    AIcrowdTransientError,
    AIcrowdValidationError,
    _classify,
    _compute_backoff,
    _parse_retry_after,
    describe_error,
    extract_submission_id,
)


def _client(handler) -> AIcrowdClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return AIcrowdClient(api_key="K", http=http)


def test_verify_identity_sends_token_header_and_returns_participant_id():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"id": 4242, "username": "alice"})

    pid = _client(handler).verify_identity()
    assert pid == 4242
    assert seen["auth"] == "Token K"
    assert seen["url"].endswith("/api/v1/api_user")


def test_verify_identity_401_raises():
    def handler(req):
        return httpx.Response(401, json={"message": "bad key"})

    with pytest.raises(AIcrowdAPIError):
        _client(handler).verify_identity()


def test_check_eligibility_queries_the_eligibility_endpoint():
    """The pre-flight must ask the endpoint that applies the real submission checks,
    not infer the answer from a registration flag."""
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(
            200,
            json={
                "submissions_allowed": False,
                "denied_reason": "challenge_completed",
                "message": "Phase 1 is completed — submissions are closed.",
                "rules_accepted": True,
                "participation_terms_accepted": True,
            },
        )

    out = _client(handler).check_eligibility(challenge_slug="some-challenge")
    assert "www.aicrowd.com/api/v1/challenges/some-challenge/eligibility" in seen["url"]
    assert "api.aicrowd.com" not in seen["url"]
    assert out["submissions_allowed"] is False
    assert out["denied_reason"] == "challenge_completed"


def test_whoami_keeps_the_username():
    """`whest login` greets the participant by name, so identity must not be
    reduced to an id on the way through."""

    def handler(req):
        return httpx.Response(200, json={"id": 4242, "username": "bob", "is_admin": False})

    ident = _client(handler).whoami()
    assert ident["username"] == "bob"
    assert ident["id"] == 4242


def test_401_after_identity_is_a_permission_error_not_an_auth_one():
    """A 401 can mean "terms not accepted", not "bad key". Once the key has been
    validated, reporting a later 401 as an auth failure sends people to re-login for
    no reason."""
    from whestbench.aicrowd_client import AIcrowdAuthError, AIcrowdNotAllowedError

    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"id": 1, "username": "bob"})
        return httpx.Response(
            401,
            json={"message": "Please accept challenge terms before making submission here: url"},
        )

    client = _client(handler)
    client.verify_identity()
    with pytest.raises(AIcrowdNotAllowedError) as exc:
        client.get_upload_details(challenge_slug="c")
    assert not isinstance(exc.value, AIcrowdAuthError)
    assert "accept challenge terms" in exc.value.message
    # AIcrowd explained itself; we must not staple a second, contradicting cause on.
    assert exc.value.hint is None


def test_401_before_identity_is_still_an_auth_error():
    def handler(req):
        return httpx.Response(401, json={"message": "bad key"})

    from whestbench.aicrowd_client import AIcrowdAuthError

    with pytest.raises(AIcrowdAuthError):
        _client(handler).get_upload_details(challenge_slug="c")


def test_403_with_a_server_explanation_adds_no_hint():
    """A closed round is not fixed by accepting anything, so the old
    "accept the rules and confirm submissions are open" tip must not appear."""
    from whestbench.aicrowd_client import AIcrowdNotAllowedError

    def handler(req):
        return httpx.Response(
            403,
            json={
                "message": "Phase 1 is completed — submissions are closed. "
                "Phase 2 opens on August 21, 2026 23:59."
            },
        )

    with pytest.raises(AIcrowdNotAllowedError) as exc:
        _client(handler).get_upload_details(challenge_slug="c")
    assert "submissions are closed" in exc.value.message
    assert exc.value.hint is None
    assert "rules" not in (exc.value.hint or "")


def test_get_upload_details_unwraps_data_and_passes_slug():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(
            200,
            json={
                "message": "Presigned key generated!",
                "success": True,
                "data": {"url": "https://s3.test/up", "fields": {"key": "subs/${filename}"}},
            },
        )

    up = _client(handler).get_upload_details(challenge_slug="slugX")
    assert "challenge_id=slugX" in seen["url"]
    assert up["url"] == "https://s3.test/up"
    assert up["fields"]["key"] == "subs/${filename}"


def test_create_submission_sends_nested_body():
    captured = {}

    def handler(req):
        captured["body"] = _json.loads(req.content.decode())
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {"submission_id": 7777, "created_at": "2026-05-29T00:00:00Z"},
            },
        )

    resp = _client(handler).create_submission(
        challenge_slug="slugX", s3_key="subs/sub.tar.gz", description="whest"
    )
    body = captured["body"]
    assert body["challenge_id"] == "slugX"
    assert body["submission"] == {"description": "whest"}
    # API requests carry a TOP-LEVEL submission_files array (the controller
    # ignores nested submission_files_attributes for is_api_request).
    assert body["submission_files"] == [{"submission_file_s3_key": "subs/sub.tar.gz"}]
    assert extract_submission_id(resp) == 7777


def test_get_submission_status():
    def handler(req):
        # Api::V1::SubmissionsController#show: participant-token route at
        # /api/v1/submissions/{id}, returning the grading_status_cd serializer.
        assert str(req.url).rstrip("/").endswith("/api/v1/submissions/7777")
        return httpx.Response(
            200,
            json={
                "id": 7777,
                "grading_status_cd": "graded",
                "grading_message": "Graded successfully",
                "score": 0.91,
            },
        )

    st = _client(handler).get_submission_status(7777)
    assert st["grading_status_cd"] == "graded"
    assert st["score"] == 0.91


def test_extract_submission_id_handles_response_shapes():
    assert extract_submission_id({"data": {"submission_id": 1}}) == 1
    assert extract_submission_id({"submission_id": 2}) == 2
    assert extract_submission_id({"id": 3}) == 3
    assert extract_submission_id({"data": {"id": 4}}) == 4
    assert extract_submission_id({"nope": 1}) is None


def test_apierror_defaults_to_non_transient():
    err = AIcrowdAPIError(status=404, message="missing")
    assert err.status == 404
    assert err.transient is False


def test_transient_error_is_apierror_subclass_and_transient():
    err = AIcrowdTransientError(status=503, message="maintenance")
    assert isinstance(err, AIcrowdAPIError)
    assert err.transient is True
    assert err.status == 503


class _MaxJitter:
    def uniform(self, lo: float, hi: float) -> float:
        return hi  # full-jitter upper bound — makes backoff deterministic


class _ZeroJitter:
    def uniform(self, lo: float, hi: float) -> float:
        return lo


def test_compute_backoff_is_exponential_and_capped():
    rng = _MaxJitter()
    assert _compute_backoff(1, retry_after=None, base=0.5, cap=8.0, rng=rng) == 0.5
    assert _compute_backoff(2, retry_after=None, base=0.5, cap=8.0, rng=rng) == 1.0
    assert _compute_backoff(5, retry_after=None, base=0.5, cap=8.0, rng=rng) == 8.0  # 0.5*16 -> cap
    assert _compute_backoff(6, retry_after=None, base=0.5, cap=8.0, rng=rng) == 8.0  # stays capped


def test_compute_backoff_jitter_lower_bound_is_zero():
    assert _compute_backoff(3, retry_after=None, base=0.5, cap=8.0, rng=_ZeroJitter()) == 0.0


def test_compute_backoff_retry_after_wins_when_larger():
    assert _compute_backoff(1, retry_after=30.0, base=0.5, cap=8.0, rng=_MaxJitter()) == 30.0


def test_parse_retry_after_seconds_and_garbage():
    assert _parse_retry_after("2") == 2.0
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("not-a-date") is None


def test_retry_presets_and_status_set():
    assert SUBMIT_RETRY.max_attempts == 5
    assert POLL_RETRY.max_attempts == 3
    assert 503 in _RETRYABLE_STATUS and 401 not in _RETRYABLE_STATUS


def test_request_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr(client_mod, "_sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="maintenance")
        return httpx.Response(200, json={"ok": True})

    r = _client(handler)._get("https://www.aicrowd.com/api/v1/x")
    assert r.status_code == 200
    assert calls["n"] == 3


def test_request_permanent_4xx_raises_immediately(monkeypatch):
    monkeypatch.setattr(client_mod, "_sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401, json={"message": "bad key"})

    with pytest.raises(AIcrowdAPIError) as ei:
        _client(handler)._get("https://www.aicrowd.com/api/v1/x")
    assert ei.value.transient is False
    assert calls["n"] == 1


def test_request_exhausts_transient_raises_transient(monkeypatch):
    monkeypatch.setattr(client_mod, "_sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, text="down")

    with pytest.raises(AIcrowdTransientError) as ei:
        _client(handler)._get("https://www.aicrowd.com/api/v1/x")
    assert ei.value.transient is True
    assert calls["n"] == 5


def test_request_retries_transport_error(monkeypatch):
    monkeypatch.setattr(client_mod, "_sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"ok": True})

    r = _client(handler)._get("https://www.aicrowd.com/api/v1/x")
    assert r.status_code == 200
    assert calls["n"] == 2


def test_request_honors_retry_after_header(monkeypatch):
    slept = []
    monkeypatch.setattr(client_mod, "_sleep", lambda s: slept.append(s))
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, headers={"Retry-After": "2"}, text="down")
        return httpx.Response(200, json={"ok": True})

    _client(handler)._get("https://www.aicrowd.com/api/v1/x")
    assert slept == [2.0]


def test_get_submission_status_retries_then_returns(monkeypatch):
    monkeypatch.setattr(client_mod, "_sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="down")
        return httpx.Response(200, json={"id": 1, "grading_status_cd": "graded"})

    st = _client(handler).get_submission_status(1)
    assert st["grading_status_cd"] == "graded"
    assert calls["n"] == 3  # POLL_RETRY succeeds on the 3rd attempt


def test_get_submission_status_exhausts_poll_budget(monkeypatch):
    monkeypatch.setattr(client_mod, "_sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, text="down")

    with pytest.raises(AIcrowdTransientError):
        _client(handler).get_submission_status(1)
    assert calls["n"] == 3  # POLL_RETRY.max_attempts (not the submit budget of 5)


def test_parse_retry_after_http_date_returns_positive_seconds():
    # A far-future GMT (tz-aware) date yields a positive number of seconds.
    secs = _parse_retry_after("Wed, 01 Jan 2200 00:00:00 GMT")
    assert isinstance(secs, float) and secs > 0


def test_parse_retry_after_naive_datetime_is_rejected():
    # A date string without a timezone is naive and must be rejected (None),
    # not used to compute a timezone-incorrect delta.
    assert _parse_retry_after("Wed, 01 Jan 2200 00:00:00") is None


def test_upload_to_s3_sends_no_token_and_retries(monkeypatch, tmp_path):
    monkeypatch.setattr(client_mod, "_sleep", lambda s: None)
    seen = {"n": 0, "auth": []}

    def handler(req):
        seen["n"] += 1
        seen["auth"].append(req.headers.get("Authorization"))
        if seen["n"] < 2:
            return httpx.Response(503, text="s3 down")
        return httpx.Response(204)

    f = tmp_path / "submission.tar.gz"
    f.write_bytes(b"payload")
    key = _client(handler).upload_to_s3(
        upload={"url": "https://s3.test/upload", "fields": {"key": "subs/${filename}"}},
        file_path=str(f),
    )
    assert key == "subs/submission.tar.gz"
    assert seen["n"] == 2  # retried the transient 503 once
    assert seen["auth"] == [None, None]  # never sends the AIcrowd token to S3


def test_error_summary_includes_op_and_status():
    e = AIcrowdAPIError(status=403, message="nope", op="creating your submission")
    assert e.summary == "While creating your submission: nope (HTTP 403)"


def test_error_summary_omits_http_for_nonpositive_status():
    e = AIcrowdAPIError(status=0, message="network down", op="uploading the artifact")
    assert e.summary == "While uploading the artifact: network down"


def test_error_summary_without_op():
    assert AIcrowdAPIError(status=404, message="missing").summary == "missing (HTTP 404)"


def test_subclasses_carry_stable_code_and_default_hint():
    assert AIcrowdAuthError(status=401, message="x").code == "auth"
    assert "whest login" in (AIcrowdAuthError(status=401, message="x").hint or "")
    assert AIcrowdNotAllowedError(status=403, message="x").code == "not_allowed"
    assert AIcrowdNotFoundError(status=404, message="x").code == "not_found"
    assert AIcrowdValidationError(status=422, message="x").code == "validation"
    assert AIcrowdTransientError(status=503, message="x").code == "transient"


def test_hierarchy_preserved():
    assert isinstance(AIcrowdAuthError(status=401, message="x"), AIcrowdAPIError)
    assert isinstance(AIcrowdTransientError(status=503, message="x"), AIcrowdAPIError)


def _resp(status, *, json=None, text=None, headers=None):
    return httpx.Response(
        status,
        json=json,
        text=text,
        headers=headers or {},
        request=httpx.Request("GET", "https://www.aicrowd.com/api/v1/x"),
    )


def test_classify_403_prefers_server_json_message():
    e = _classify(
        _resp(403, json={"error": "Submissions are not open.", "success": False}),
        op="creating your submission",
    )
    assert isinstance(e, AIcrowdNotAllowedError)
    assert e.message == "Submissions are not open."
    assert e.op == "creating your submission"


def test_classify_401_is_auth_error():
    assert isinstance(_classify(_resp(401, json={"error": "bad key"})), AIcrowdAuthError)


def test_classify_redirect_is_auth_error_and_discards_html():
    e = _classify(
        _resp(
            302,
            text="<html><body>You are being <a href='/'>redirected</a>.</body></html>",
            headers={"location": "https://www.aicrowd.com/"},
        ),
        op="creating your submission",
    )
    assert isinstance(e, AIcrowdAuthError)
    assert "<html>" not in e.message and "<a" not in e.message
    assert e.status == 302


def test_classify_html_body_is_discarded_for_non_json():
    e = _classify(_resp(403, text="<html>forbidden</html>"))
    assert "<html>" not in e.message
    assert e.message  # a per-status default, not empty


def test_classify_404_and_422():
    assert isinstance(_classify(_resp(404, json={"message": "no challenge"})), AIcrowdNotFoundError)
    assert isinstance(_classify(_resp(422, json={"message": "bad file"})), AIcrowdValidationError)


def test_describe_error_for_typed_and_generic():
    typed = describe_error(
        AIcrowdNotAllowedError(status=403, message="nope", op="creating your submission")
    )
    assert typed["code"] == "not_allowed" and typed["status"] == 403
    assert typed["message"] == "While creating your submission: nope (HTTP 403)"
    assert typed["hint"]
    generic = describe_error(ValueError("boom"))
    assert generic == {"message": "boom", "hint": None, "code": "error", "status": None}


def test_create_submission_403_raises_not_allowed_with_op():
    def handler(req):
        return httpx.Response(403, json={"error": "Submissions are not open.", "success": False})

    with pytest.raises(AIcrowdNotAllowedError) as ei:
        _client(handler).create_submission(challenge_slug="c", s3_key="k", description="d")
    assert ei.value.op == "creating your submission"
    assert ei.value.message == "Submissions are not open."


def test_create_submission_redirect_raises_auth_error():
    def handler(req):
        return httpx.Response(
            302,
            headers={"location": "https://www.aicrowd.com/"},
            text="<html>You are being redirected.</html>",
        )

    with pytest.raises(AIcrowdAuthError):
        _client(handler).create_submission(challenge_slug="c", s3_key="k", description="d")


def test_verify_identity_401_is_auth_error_with_op():
    def handler(req):
        return httpx.Response(401, json={"error": "bad key"})

    with pytest.raises(AIcrowdAuthError) as ei:
        _client(handler).verify_identity()
    assert ei.value.op == "verifying your API key"


def test_transient_exhaustion_carries_op_and_no_html(monkeypatch):
    monkeypatch.setattr(client_mod, "_sleep", lambda *_: None)

    def handler(req):
        return httpx.Response(503, text="<html>maintenance</html>")

    with pytest.raises(AIcrowdTransientError) as ei:
        _client(handler).create_submission(challenge_slug="c", s3_key="k", description="d")
    assert ei.value.op == "creating your submission"
    assert "<html>" not in ei.value.message
