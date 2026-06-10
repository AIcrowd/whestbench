"""Submission size/count caps. Import-light (no flopscope/datasets) so both
``whest package`` and the evaluator's constrained ingestion venv import it cheaply.
Single source of truth — the participant-facing promise and the evaluator-side guard."""

from __future__ import annotations

MAX_SUBMISSION_BYTES: int = 50 * 1024 * 1024  # 50 MiB total (sum of bundled files)
MAX_SUBMISSION_FILES: int = 50  # max files in a submission
