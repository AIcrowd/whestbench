"""Thin AIcrowd REST client for `whest submit` (hop A only).

Contract VERIFIED against the AIcrowd Rails source (app/controllers/
submissions_controller.rb + app/controllers/api/v1/submissions_controller.rb +
api/v1/api_users_controller.rb + base_controller.rb):

- Auth header: `Authorization: Token <api_key>`  (NOT Bearer)
- Rails API base: https://www.aicrowd.com/api/v1   (RAILS_HOST env overrides host)
- AIcrowd API base: https://api.aicrowd.com         (AICROWD_API_ENDPOINT env overrides)
- Identity:     GET  {rails}/api_user                          -> {"id": <participant_id>, ...}
- Challenge id: GET  {aicrowd}/challenges/?slug=...            -> [{"id": ..., "slug": ...}]
- Registration: GET  {aicrowd}/challenges/{id}/participant?participant_id=<id>
                                                               -> {"registered": bool}
- Presign:      GET  {rails}/submissions?challenge_id=<slug>   -> {"data": {"fields": {...}, "url": ...}, "success": true}
- S3 upload:    multipart POST to data.url with data.fields; substitute ${filename} in fields["key"].
- Create:       POST {rails}/submissions  (handle_artifact_based_submissions, is_api_request:
                resets submission_files_attributes and reads a TOP-LEVEL `submission_files`
                array, setting submission_type='artifact' itself):
                  {"challenge_id": "<slug>",
                   "submission": {"description": ...},
                   "submission_files": [{"submission_file_s3_key": "<key>"}]}
                -> {"data": {"submission_id": <id>, "created_at": ...}, "success": true}
- Status:       GET  {rails}/submissions/{id}  (Api::V1::SubmissionsController#show;
                participant-token auth, authorized to the caller's own submission)
                -> {..., "grading_status_cd": "ready"|"submitted"|"initiated"|"graded"|"failed",
                    "score": ..., "score_secondary": ..., "grading_message": ...}.
                Transient failures (429/5xx/network) are retried with bounded
                backoff inside this client; the --watch loop additionally rides
                out blips silently until a terminal state or its --watch-timeout,
                so a poll failure never turns a successful submit into a failure.
"""

from __future__ import annotations

import datetime
import json as _jsonlib
import os
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional, Protocol

import httpx


def _rails_base() -> str:
    host = os.environ.get("RAILS_HOST", "www.aicrowd.com")
    return f"https://{host}/api/v1"


def _aicrowd_base() -> str:
    return os.environ.get("AICROWD_API_ENDPOINT", "https://api.aicrowd.com")


# --- retry layer (transient-error resilience) ----------------------------
# Status codes worth retrying: rate-limit + transient server/proxy errors.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Injectable seams so tests never actually sleep and jitter is deterministic.
_sleep = time.sleep
_monotonic = time.monotonic
_rng = random.Random()


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry budget for one logical API call."""

    max_attempts: int
    base_delay: float
    max_delay: float
    deadline_s: Optional[float] = None


# Submit path: user is blocked, so retry briefly then surface a real error.
SUBMIT_RETRY = RetryPolicy(max_attempts=5, base_delay=0.5, max_delay=8.0, deadline_s=45.0)
# Watch poll: light per-call retry; the --watch loop supplies the real patience.
POLL_RETRY = RetryPolicy(max_attempts=3, base_delay=0.5, max_delay=4.0, deadline_s=20.0)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a Retry-After header to seconds. Supports the integer-seconds form
    and the (timezone-aware) HTTP-date form; returns None on absence, garbage,
    or a timezone-naive date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        return None
    now = datetime.datetime.now(tz=dt.tzinfo)
    return max(0.0, (dt - now).total_seconds())


class _RandomLike(Protocol):
    """Minimal RNG surface used by ``_compute_backoff`` (satisfied by ``random.Random``)."""

    def uniform(self, a: float, b: float, /) -> float: ...


def _compute_backoff(
    attempt: int,
    *,
    retry_after: Optional[float],
    base: float,
    cap: float,
    rng: "_RandomLike",
) -> float:
    """1-based attempt. Exponential base*2**(attempt-1), capped at `cap`, with
    full jitter in [0, exp]. An explicit, larger Retry-After wins (no jitter)."""
    exp = min(cap, base * (2 ** (attempt - 1)))
    delay = rng.uniform(0.0, exp)
    if retry_after is not None and retry_after > delay:
        return retry_after
    return delay


def extract_submission_id(resp: dict[str, Any]) -> Optional[int]:
    """Pull the submission id out of a create/response payload, tolerating the
    `data`-wrapper and either `submission_id` or `id` keys."""
    for container in (resp.get("data") if isinstance(resp.get("data"), dict) else None, resp):
        if not isinstance(container, dict):
            continue
        for key in ("submission_id", "id"):
            val = container.get(key)
            if val is not None:
                return int(val)
    return None


class AIcrowdAPIError(RuntimeError):
    """A non-2xx (or redirect) from an AIcrowd endpoint, rendered for humans."""

    def __init__(
        self,
        *,
        status: int,
        message: str,
        hint: Optional[str] = None,
        code: str = "api_error",
        op: Optional[str] = None,
        transient: bool = False,
    ) -> None:
        self.status = status
        self.message = message
        self.hint = hint
        self.code = code
        self.op = op
        self.transient = transient
        super().__init__(self.summary)

    @property
    def summary(self) -> str:
        prefix = f"While {self.op}: " if self.op else ""
        suffix = f" (HTTP {self.status})" if self.status and self.status > 0 else ""
        return f"{prefix}{self.message}{suffix}"


class AIcrowdAuthError(AIcrowdAPIError):
    """401, or a redirect to a web login page — missing/invalid API key, or not authorized."""

    def __init__(self, *, status: int, message: str, op: Optional[str] = None) -> None:
        super().__init__(
            status=status,
            message=message,
            code="auth",
            op=op,
            hint="Run `whest login` (copy your API key from your AIcrowd profile page).",
        )


class AIcrowdNotAllowedError(AIcrowdAPIError):
    """403 — not authorized for this action (e.g. submissions not open, rules not accepted)."""

    def __init__(self, *, status: int, message: str, op: Optional[str] = None) -> None:
        super().__init__(
            status=status,
            message=message,
            code="not_allowed",
            op=op,
            hint="Open the challenge page: accept the rules and confirm submissions are open.",
        )


class AIcrowdNotFoundError(AIcrowdAPIError):
    """404 — the challenge or endpoint was not found."""

    def __init__(self, *, status: int, message: str, op: Optional[str] = None) -> None:
        super().__init__(
            status=status,
            message=message,
            code="not_found",
            op=op,
            hint="Check the challenge slug (e.g. arc-white-box-estimation-challenge-2026).",
        )


class AIcrowdValidationError(AIcrowdAPIError):
    """400/422 — the request/submission was rejected."""

    def __init__(self, *, status: int, message: str, op: Optional[str] = None) -> None:
        super().__init__(
            status=status,
            message=message,
            code="validation",
            op=op,
            hint="Check your submission file and metadata, then try again.",
        )


class AIcrowdTransientError(AIcrowdAPIError):
    """A retryable failure (429/5xx/network) that exhausted its retry budget."""

    def __init__(self, *, status: int, message: str, op: Optional[str] = None) -> None:
        super().__init__(status=status, message=message, code="transient", op=op, transient=True)


_STATUS_DEFAULTS = {
    400: "The request was rejected.",
    401: "Your AIcrowd API key is missing or invalid.",
    403: "You're not authorized to perform this action — submissions may not be open for "
         "this challenge, or you may need to accept the challenge rules.",
    404: "Not found.",
    422: "Your submission was rejected.",
}

_REDIRECT_MESSAGE = (
    "The AIcrowd API redirected to a web page instead of returning data — your API key is "
    "likely invalid, or you're not authorized to perform this action."
)


def _server_message(response: "httpx.Response") -> Optional[str]:
    """Return a human message from a JSON error body ({"error"|"message": ...}), else None.
    HTML and non-JSON bodies are intentionally ignored so they never reach the user."""
    ctype = response.headers.get("content-type", "")
    if "application/json" not in ctype:
        return None
    try:
        data = response.json()
    except (ValueError, _jsonlib.JSONDecodeError):
        return None
    if isinstance(data, dict):
        for key in ("error", "message"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _classify(response: "httpx.Response", *, op: Optional[str] = None) -> AIcrowdAPIError:
    """Map an AIcrowd HTTP response to a typed, human-readable error.
    Prefers the server's JSON message; never surfaces HTML; redirects → auth error."""
    status = response.status_code
    if 300 <= status < 400:
        return AIcrowdAuthError(status=status, message=_REDIRECT_MESSAGE, op=op)
    message = _server_message(response) or _STATUS_DEFAULTS.get(status) or (
        f"Unexpected response from AIcrowd (HTTP {status})."
    )
    if status == 401:
        return AIcrowdAuthError(status=status, message=message, op=op)
    if status == 403:
        return AIcrowdNotAllowedError(status=status, message=message, op=op)
    if status == 404:
        return AIcrowdNotFoundError(status=status, message=message, op=op)
    if status in (400, 422):
        return AIcrowdValidationError(status=status, message=message, op=op)
    return AIcrowdAPIError(status=status, message=message, op=op)


def describe_error(exc: BaseException) -> dict:
    """Uniform fields for surfacing any error (CLI text + --json), for typed and generic errors."""
    if isinstance(exc, AIcrowdAPIError):
        return {"message": exc.summary, "hint": exc.hint, "code": exc.code, "status": exc.status}
    return {"message": str(exc), "hint": None, "code": "error", "status": None}


class AIcrowdClient:
    def __init__(
        self,
        *,
        api_key: str,
        http: Optional[httpx.Client] = None,
        timeout: float = 60.0,
    ) -> None:
        self._key = api_key
        self._http = http or httpx.Client(timeout=timeout, follow_redirects=False)
        self._auth = {"Authorization": f"Token {api_key}"}

    # --- helpers ----------------------------------------------------------
    def _request(
        self,
        method: str,
        url: str,
        *,
        policy: RetryPolicy = SUBMIT_RETRY,
        auth: bool = True,
        op: Optional[str] = None,
        **kw,
    ) -> httpx.Response:
        """Issue one logical request, retrying transient failures (429/5xx and httpx
        transport errors) within `policy`'s budget. Permanent non-2xx (and redirects)
        raise a typed, human-readable error via `_classify`. `auth=False` omits the
        AIcrowd token (for the presigned S3 upload to a different host)."""
        headers = self._auth if auth else None
        deadline = _monotonic() + policy.deadline_s if policy.deadline_s is not None else None
        last_exc: Optional[AIcrowdTransientError] = None
        for attempt in range(1, policy.max_attempts + 1):
            retry_after: Optional[float] = None
            try:
                r = self._http.request(method, url, headers=headers, **kw)
            except httpx.TransportError as e:
                last_exc = AIcrowdTransientError(
                    status=0,
                    message=f"Could not reach AIcrowd ({type(e).__name__}).",
                    op=op,
                )
            else:
                if r.is_success:
                    return r
                if r.status_code in _RETRYABLE_STATUS:
                    last_exc = AIcrowdTransientError(
                        status=r.status_code,
                        message=_server_message(r) or "AIcrowd is temporarily unavailable.",
                        op=op,
                    )
                    retry_after = _parse_retry_after(r.headers.get("Retry-After"))
                else:
                    raise _classify(r, op=op)
            if attempt >= policy.max_attempts:
                break
            delay = _compute_backoff(
                attempt,
                retry_after=retry_after,
                base=policy.base_delay,
                cap=policy.max_delay,
                rng=_rng,
            )
            if deadline is not None and _monotonic() + delay >= deadline:
                break
            _sleep(delay)
        assert last_exc is not None  # loop only exits early via return/raise above
        raise last_exc

    def _get(self, url: str, *, policy: RetryPolicy = SUBMIT_RETRY, op: Optional[str] = None, **kw) -> httpx.Response:
        return self._request("GET", url, policy=policy, op=op, **kw)

    def _post(self, url: str, *, policy: RetryPolicy = SUBMIT_RETRY, op: Optional[str] = None, **kw) -> httpx.Response:
        return self._request("POST", url, policy=policy, op=op, **kw)

    # --- identity + challenge --------------------------------------------
    def verify_identity(self) -> int:
        """Validate the key; return the participant id."""
        return int(self._get(f"{_rails_base()}/api_user", op="verifying your API key").json()["id"])

    def resolve_challenge(self, slug: str) -> int:
        """Resolve a challenge slug -> numeric challenge id (for the registration check)."""
        r = self._get(f"{_aicrowd_base()}/challenges/", params={"slug": slug}, op="resolving the challenge")
        data = r.json()
        items = data if isinstance(data, list) else data.get("data", [])
        for item in items:
            if item.get("slug") == slug:
                return int(item["id"])
        if items:
            return int(items[0]["id"])
        raise AIcrowdAPIError(status=404, message=f"challenge not found: {slug}")

    def check_registration(self, *, challenge_id: int, participant_id: int) -> bool:
        r = self._get(
            f"{_aicrowd_base()}/challenges/{challenge_id}/participant",
            params={"participant_id": participant_id},
            op="checking your challenge registration",
        )
        return bool(r.json().get("registered"))

    # --- submission upload + create --------------------------------------
    def get_upload_details(self, *, challenge_slug: str) -> dict[str, Any]:
        """Presigned S3 POST details: {"url": ..., "fields": {...}}."""
        r = self._get(
            f"{_rails_base()}/submissions",
            params={"challenge_id": challenge_slug},
            op="preparing the upload",
        )
        data = r.json()
        return data.get("data", data)

    def upload_to_s3(self, *, upload: dict[str, Any], file_path: str) -> str:
        """Multipart POST the artifact to S3; return the resulting object key.

        AIcrowd's presigned POST returns fields where `key` contains a
        `${filename}` placeholder S3 substitutes with the uploaded filename.
        We substitute it locally too so we can report the final key to Rails.
        The body is read into memory (artifacts are ≤50 MB, typically a few KB)
        so it is re-sendable across retries; `auth=False` keeps the AIcrowd token
        off the S3 request."""
        fields = dict(upload["fields"])
        fname = Path(file_path).name
        s3_key = fields.get("key", "").replace("${filename}", fname)
        fields["key"] = s3_key
        content = Path(file_path).read_bytes()
        self._request(
            "POST",
            upload["url"],
            policy=SUBMIT_RETRY,
            auth=False,
            op="uploading the artifact",
            data=fields,
            files={"file": (fname, content)},
        )
        return s3_key

    def create_submission(
        self, *, challenge_slug: str, s3_key: str, description: str
    ) -> dict[str, Any]:
        """Create the submission. challenge_id is the SLUG (Rails set_challenge
        resolves params[:challenge_id]). For API requests the controller reads a
        TOP-LEVEL `submission_files` array (it ignores any nested
        submission_files_attributes and sets submission_type='artifact' itself)."""
        r = self._post(
            f"{_rails_base()}/submissions",
            op="creating your submission",
            json={
                "challenge_id": challenge_slug,
                "submission": {"description": description},
                "submission_files": [{"submission_file_s3_key": s3_key}],
            },
        )
        return r.json()

    def get_submission_status(self, submission_id: int) -> dict[str, Any]:
        """Fetch a single submission's grading state (Api::SubmissionSerializer):
        {"grading_status_cd": ..., "score": ..., "grading_message": ..., ...}.
        Uses the lighter POLL_RETRY budget; the --watch loop supplies patience."""
        return self._get(
            f"{_rails_base()}/submissions/{submission_id}", policy=POLL_RETRY, op="checking submission status"
        ).json()
