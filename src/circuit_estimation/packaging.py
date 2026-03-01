"""Submission packaging helpers for participant estimator artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .loader import load_estimator_from_path


@dataclass(frozen=True, slots=True)
class SubmissionFiles:
    estimator: Path
    requirements: Path | None = None
    submission_yaml: Path | None = None
    approach_md: Path | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    class_name: str,
    files: SubmissionFiles,
    packager_version: str = "0.1.0",
) -> dict[str, Any]:
    included_files: list[tuple[str, Path]] = [("estimator.py", files.estimator)]
    if files.requirements is not None:
        included_files.append(("requirements.txt", files.requirements))
    if files.submission_yaml is not None:
        included_files.append(("submission.yaml", files.submission_yaml))
    if files.approach_md is not None:
        included_files.append(("APPROACH.md", files.approach_md))

    manifest_files = [
        {
            "name": arcname,
            "sha256": _sha256(path),
        }
        for arcname, path in included_files
    ]
    return {
        "schema_version": "1.0",
        "api_version": "1.0",
        "entrypoint": {"module": "estimator", "class": class_name},
        "python": {"min_version": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}"},
        "files": manifest_files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "packager_version": packager_version,
    }


def package_submission(
    estimator_path: str | Path,
    *,
    class_name: str | None = None,
    requirements_path: str | Path | None = None,
    submission_yaml_path: str | Path | None = None,
    approach_md_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    estimator = Path(estimator_path).resolve()
    if not estimator.is_file():
        raise FileNotFoundError(f"Estimator file not found: {estimator}")
    # Resolve and validate class entrypoint before packing.
    _, metadata = load_estimator_from_path(estimator, class_name=class_name)

    requirements = Path(requirements_path).resolve() if requirements_path is not None else None
    submission_yaml = (
        Path(submission_yaml_path).resolve() if submission_yaml_path is not None else None
    )
    approach_md = Path(approach_md_path).resolve() if approach_md_path is not None else None
    files = SubmissionFiles(
        estimator=estimator,
        requirements=requirements if requirements and requirements.is_file() else None,
        submission_yaml=submission_yaml if submission_yaml and submission_yaml.is_file() else None,
        approach_md=approach_md if approach_md and approach_md.is_file() else None,
    )
    manifest = build_manifest(class_name=metadata.class_name, files=files)
    manifest_blob = json.dumps(manifest, indent=2).encode("utf-8")

    target = Path(output_path).resolve() if output_path is not None else (
        Path.cwd()
        / f"submission-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.tar.gz"
    )
    with tarfile.open(target, mode="w:gz") as archive:
        archive.add(estimator, arcname="estimator.py")
        if files.requirements is not None:
            archive.add(files.requirements, arcname="requirements.txt")
        if files.submission_yaml is not None:
            archive.add(files.submission_yaml, arcname="submission.yaml")
        if files.approach_md is not None:
            archive.add(files.approach_md, arcname="APPROACH.md")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_blob)
        info.mtime = datetime.now(timezone.utc).timestamp()
        archive.addfile(info, fileobj=_bytes_io(manifest_blob))
    return target


def _bytes_io(payload: bytes):
    from io import BytesIO

    return BytesIO(payload)
