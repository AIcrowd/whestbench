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
                The --watch loop keeps a best-effort try/except so a poll failure never
                turns a successful submit into a failure.
"""

from __future__ import annotations

import datetime
import os
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional

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


def _compute_backoff(
    attempt: int,
    *,
    retry_after: Optional[float],
    base: float,
    cap: float,
    rng: "random.Random",
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
    """Non-2xx from an AIcrowd endpoint."""

    def __init__(self, *, status: int, message: str, transient: bool = False) -> None:
        super().__init__(f"AIcrowd API error ({status}): {message}")
        self.status = status
        self.message = message
        self.transient = transient


class AIcrowdTransientError(AIcrowdAPIError):
    """A retryable failure (429/5xx/network) that exhausted its retry budget."""

    def __init__(self, *, status: int, message: str) -> None:
        super().__init__(status=status, message=message, transient=True)


class AIcrowdClient:
    def __init__(
        self,
        *,
        api_key: str,
        http: Optional[httpx.Client] = None,
        timeout: float = 60.0,
    ) -> None:
        self._key = api_key
        self._http = http or httpx.Client(timeout=timeout)
        self._auth = {"Authorization": f"Token {api_key}"}

    # --- helpers ----------------------------------------------------------
    def _request(self, method: str, url: str, *, policy: RetryPolicy = SUBMIT_RETRY,
                 auth: bool = True, **kw) -> httpx.Response:
        """Issue one logical request, retrying transient failures (429/5xx and
        httpx transport errors) within `policy`'s budget. Permanent non-2xx
        (e.g. 401/403/404) raise immediately. `auth=False` omits the AIcrowd
        token (for the presigned S3 upload to a different host)."""
        headers = self._auth if auth else None
        deadline = _monotonic() + policy.deadline_s if policy.deadline_s is not None else None
        last_exc: Optional[AIcrowdTransientError] = None
        for attempt in range(1, policy.max_attempts + 1):
            retry_after: Optional[float] = None
            try:
                r = self._http.request(method, url, headers=headers, **kw)
            except httpx.TransportError as e:
                last_exc = AIcrowdTransientError(status=0, message=f"{type(e).__name__}: {e}")
            else:
                if r.is_success:
                    return r
                if r.status_code in _RETRYABLE_STATUS:
                    last_exc = AIcrowdTransientError(status=r.status_code, message=r.text[:300])
                    retry_after = _parse_retry_after(r.headers.get("Retry-After"))
                else:
                    raise AIcrowdAPIError(status=r.status_code, message=r.text[:300])
            if attempt >= policy.max_attempts:
                break
            delay = _compute_backoff(
                attempt, retry_after=retry_after,
                base=policy.base_delay, cap=policy.max_delay, rng=_rng,
            )
            if deadline is not None and _monotonic() + delay >= deadline:
                break
            _sleep(delay)
        assert last_exc is not None  # loop only exits early via return/raise above
        raise last_exc

    def _get(self, url: str, *, policy: RetryPolicy = SUBMIT_RETRY, **kw) -> httpx.Response:
        return self._request("GET", url, policy=policy, **kw)

    def _post(self, url: str, *, policy: RetryPolicy = SUBMIT_RETRY, **kw) -> httpx.Response:
        return self._request("POST", url, policy=policy, **kw)

    # --- identity + challenge --------------------------------------------
    def verify_identity(self) -> int:
        """Validate the key; return the participant id."""
        return int(self._get(f"{_rails_base()}/api_user").json()["id"])

    def resolve_challenge(self, slug: str) -> int:
        """Resolve a challenge slug -> numeric challenge id (for the registration check)."""
        r = self._get(f"{_aicrowd_base()}/challenges/", params={"slug": slug})
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
        )
        return bool(r.json().get("registered"))

    # --- submission upload + create --------------------------------------
    def get_upload_details(self, *, challenge_slug: str) -> dict[str, Any]:
        """Presigned S3 POST details: {"url": ..., "fields": {...}}."""
        r = self._get(f"{_rails_base()}/submissions", params={"challenge_id": challenge_slug})
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
            "POST", upload["url"], policy=SUBMIT_RETRY, auth=False,
            data=fields, files={"file": (fname, content)},
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
            f"{_rails_base()}/submissions/{submission_id}", policy=POLL_RETRY
        ).json()
