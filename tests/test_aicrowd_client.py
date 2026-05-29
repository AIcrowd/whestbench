"""AIcrowd REST client — call-sequence tests via httpx MockTransport.

Pins the contract verified against the AIcrowd Rails source: Token auth, the
presigned-POST `data.{url,fields}` shape, and the NESTED submission-create body
(`submission.submission_files_attributes[].submission_file_s3_key`).
"""

from __future__ import annotations

import json as _json

import httpx
import pytest

from whestbench.aicrowd_client import AIcrowdAPIError, AIcrowdClient, extract_submission_id


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


def test_resolve_challenge_returns_id():
    def handler(req):
        assert "/challenges/" in str(req.url)
        return httpx.Response(
            200, json=[{"id": 99, "slug": "arc-white-box-estimation-challenge-2026"}]
        )

    cid = _client(handler).resolve_challenge("arc-white-box-estimation-challenge-2026")
    assert cid == 99


def test_check_registration_true():
    def handler(req):
        return httpx.Response(200, json={"registered": True})

    assert _client(handler).check_registration(challenge_id=99, participant_id=4242) is True


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
        assert str(req.url).rstrip("/").endswith("/submissions/7777")
        return httpx.Response(200, json={"id": 7777, "grading_status": "graded", "score": 0.91})

    st = _client(handler).get_submission_status(7777)
    assert st["grading_status"] == "graded"


def test_extract_submission_id_handles_response_shapes():
    assert extract_submission_id({"data": {"submission_id": 1}}) == 1
    assert extract_submission_id({"submission_id": 2}) == 2
    assert extract_submission_id({"id": 3}) == 3
    assert extract_submission_id({"data": {"id": 4}}) == 4
    assert extract_submission_id({"nope": 1}) is None
