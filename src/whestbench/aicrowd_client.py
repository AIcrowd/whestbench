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

import os
from pathlib import Path
from typing import Any, Optional

import httpx


def _rails_base() -> str:
    host = os.environ.get("RAILS_HOST", "www.aicrowd.com")
    return f"https://{host}/api/v1"


def _aicrowd_base() -> str:
    return os.environ.get("AICROWD_API_ENDPOINT", "https://api.aicrowd.com")


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

    def __init__(self, *, status: int, message: str) -> None:
        super().__init__(f"AIcrowd API error ({status}): {message}")
        self.status = status
        self.message = message


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
    def _get(self, url: str, **kw) -> httpx.Response:
        r = self._http.get(url, headers=self._auth, **kw)
        if not r.is_success:
            raise AIcrowdAPIError(status=r.status_code, message=r.text[:300])
        return r

    def _post(self, url: str, **kw) -> httpx.Response:
        r = self._http.post(url, headers=self._auth, **kw)
        if not r.is_success:
            raise AIcrowdAPIError(status=r.status_code, message=r.text[:300])
        return r

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
        """
        fields = dict(upload["fields"])
        fname = Path(file_path).name
        s3_key = fields.get("key", "").replace("${filename}", fname)
        fields["key"] = s3_key
        with open(file_path, "rb") as fh:
            r = self._http.post(upload["url"], data=fields, files={"file": (fname, fh)})
        if not r.is_success:
            raise AIcrowdAPIError(status=r.status_code, message=r.text[:300])
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
        {"grading_status_cd": ..., "score": ..., "grading_message": ..., ...}."""
        return self._get(f"{_rails_base()}/submissions/{submission_id}").json()
