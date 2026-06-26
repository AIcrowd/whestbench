"""Validate a packaged submission archive against its manifest.

Before running anything, the grader unpacks a submission .tar.gz and verifies it
against the bundled manifest.json: the entrypoint module must be present, and every
files[] entry must be a regular file whose sha256 matches the archive bytes. A
directory entry (prod whestbench#107) crashes that check with IsADirectoryError.
This module runs the same verification LOCALLY so a bad archive — including a
hand-rolled or drifted one passed to ``whest submit <tarball>`` — fails fast with an
actionable message instead of crashing the grader.

It is intentionally STRICTER than the (tolerant) grader: a directory entry is
rejected here and the participant is told to ship the subpackage as individual files
(which ``whest package`` does automatically). Passing validate_package() therefore
means the archive is both well-formed and grade-safe.

TODO(evaluator-parity): align accepted versions + any member-coverage check with the
current whestbench-evaluator/worker/ingestion.py.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

_SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})
_SUPPORTED_API_VERSIONS = frozenset({"2.0"})


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    name: str  # offending files[] name / member; "" for archive-level issues
    message: str


@dataclass(frozen=True)
class PackageValidation:
    ok: bool
    issues: "List[ValidationIssue]"


def _directory_issue(name: str) -> ValidationIssue:
    return ValidationIssue(
        "directory_entry",
        name,
        f"manifest files[] entry {name!r} is a directory. A directory cannot be "
        f"hashed and crashes the grader's integrity check. Ship the subpackage as "
        f"its individual files — `whest package` lists them automatically; do not "
        f"hand-write manifest.json.",
    )


def validate_package(tarball: "str | Path") -> PackageValidation:
    """Verify a submission ``.tar.gz`` against its bundled ``manifest.json``.

    Returns a :class:`PackageValidation`; ``ok`` is True only when there are no
    issues. All problems are collected (not just the first) so one run surfaces
    everything that needs fixing."""
    path = Path(tarball)
    if not path.is_file():
        return PackageValidation(
            False,
            [ValidationIssue("archive_missing", str(path), f"No such submission archive: {path}.")],
        )
    try:
        archive = tarfile.open(path, "r:gz")
    except (tarfile.TarError, OSError) as e:
        return PackageValidation(
            False,
            [
                ValidationIssue(
                    "archive_unreadable",
                    str(path),
                    f"Cannot open {path} as a gzip tar archive: {e}.",
                )
            ],
        )

    issues: "List[ValidationIssue]" = []
    with archive:
        members = {m.name: m for m in archive.getmembers()}

        manifest_member = members.get("manifest.json")
        if manifest_member is None or not manifest_member.isreg():
            return PackageValidation(
                False,
                [
                    ValidationIssue(
                        "missing_manifest",
                        "manifest.json",
                        "Archive has no readable manifest.json (missing or not a regular file). "
                        "Build the archive with `whest package`.",
                    )
                ],
            )
        try:
            manifest = json.loads(archive.extractfile(manifest_member).read())  # type: ignore[union-attr]
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return PackageValidation(
                False,
                [
                    ValidationIssue(
                        "invalid_manifest_json",
                        "manifest.json",
                        f"manifest.json is not valid JSON: {e}.",
                    )
                ],
            )

        if not isinstance(manifest, dict):
            return PackageValidation(
                False,
                [
                    ValidationIssue(
                        "invalid_manifest_json",
                        "manifest.json",
                        f"manifest.json must be a JSON object, not a {type(manifest).__name__}. "
                        "Build the archive with `whest package`.",
                    )
                ],
            )

        schema_version = manifest.get("schema_version")
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            issues.append(
                ValidationIssue(
                    "unsupported_schema_version",
                    "manifest.json",
                    f"manifest schema_version {schema_version!r} is not supported "
                    f"(expected one of {sorted(_SUPPORTED_SCHEMA_VERSIONS)}).",
                )
            )
        api_version = manifest.get("api_version")
        if api_version not in _SUPPORTED_API_VERSIONS:
            issues.append(
                ValidationIssue(
                    "unsupported_api_version",
                    "manifest.json",
                    f"manifest api_version {api_version!r} is not supported "
                    f"(expected one of {sorted(_SUPPORTED_API_VERSIONS)}).",
                )
            )

        entrypoint = manifest.get("entrypoint")
        if not isinstance(entrypoint, dict):
            entrypoint = {}
        module = entrypoint.get("module")
        if not module:
            issues.append(
                ValidationIssue(
                    "missing_entrypoint", "manifest.json", "manifest entrypoint.module is missing."
                )
            )
        else:
            entry_file = f"{module}.py"
            m = members.get(entry_file)
            if m is None or not m.isreg():
                issues.append(
                    ValidationIssue(
                        "missing_entrypoint_file",
                        entry_file,
                        f"manifest declares entrypoint module {module!r} but the archive "
                        f"has no regular file {entry_file}.",
                    )
                )

        files = manifest.get("files")
        if not isinstance(files, list):
            issues.append(
                ValidationIssue(
                    "invalid_files", "manifest.json", "manifest files[] is missing or not a list."
                )
            )
            files = []

        for entry in files:
            name = entry.get("name", "") if isinstance(entry, dict) else ""
            declared = entry.get("sha256") if isinstance(entry, dict) else None
            if not name:
                issues.append(
                    ValidationIssue(
                        "invalid_files_entry",
                        "",
                        f"manifest files[] has an entry with no name: {entry!r}.",
                    )
                )
                continue
            if name.endswith("/"):
                issues.append(_directory_issue(name))
                continue
            m = members.get(name)
            if m is None:
                issues.append(
                    ValidationIssue(
                        "file_not_in_archive",
                        name,
                        f"manifest lists {name!r} but the archive does not contain it.",
                    )
                )
                continue
            if m.isdir():
                issues.append(_directory_issue(name))
                continue
            if not m.isreg():
                issues.append(
                    ValidationIssue(
                        "not_a_regular_file",
                        name,
                        f"manifest entry {name!r} is not a regular file in the archive "
                        f"(symlink or special file).",
                    )
                )
                continue
            handle = archive.extractfile(m)
            if handle is None:  # defensive: isreg() above guarantees a readable body
                issues.append(
                    ValidationIssue(
                        "not_a_regular_file",
                        name,
                        f"manifest entry {name!r} could not be read from the archive.",
                    )
                )
                continue
            digest = hashlib.sha256()
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
            actual = digest.hexdigest()
            if declared != actual:
                issues.append(
                    ValidationIssue(
                        "sha256_mismatch",
                        name,
                        f"sha256 mismatch for {name!r}: manifest says {declared}, archive "
                        f"bytes hash to {actual}. The file changed after the manifest was "
                        f"generated — re-run `whest package`.",
                    )
                )

    return PackageValidation(not issues, issues)
