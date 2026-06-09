"""Submission packaging helpers for participant estimator artifacts."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import platform
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import limits
from .loader import load_estimator_from_path


def _installed_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class SubmissionFiles:
    estimator: Path
    requirements: Optional[Path] = None
    submission_yaml: Optional[Path] = None
    approach_md: Optional[Path] = None


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
) -> Dict[str, Any]:
    included_files: List["tuple[str, Path]"] = [("estimator.py", files.estimator)]
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
        "api_version": "2.0",
        "entrypoint": {"module": "estimator", "class": class_name},
        "python": {
            "min_version": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}"
        },
        "files": manifest_files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "packager_version": packager_version,
        "whestbench_version": _installed_version("whestbench"),
        "flopscope_version": _installed_version("flopscope"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


class _CountingWriter:
    """File-like wrapper that reports byte counts via a callback.

    Wrapped around the raw file passed to :func:`tarfile.open` so the
    ``progress`` callback fires with the gzipped, post-compression byte count
    actually written to disk. That matches what we surface in the ``ok`` line
    after packaging completes, so the progress bar's completion count and the
    final on-disk size agree.
    """

    def __init__(self, inner: Any, cb: Callable[[int], None]) -> None:
        self._inner = inner
        self._cb = cb

    def write(self, data: bytes) -> int:
        n = self._inner.write(data)
        # Some inner writers (e.g. GzipFile) return ``None``; fall back to
        # ``len(data)``. Avoid blowing up if `n` is briefly ``None``.
        reported = n if isinstance(n, int) else len(data)
        if reported:
            self._cb(reported)
        return reported if isinstance(n, int) else n  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def package_submission(
    estimator_path: "Any",
    *,
    class_name: Optional[str] = None,
    requirements_path: "Any" = None,
    submission_yaml_path: "Any" = None,
    approach_md_path: "Any" = None,
    output_path: "Any" = None,
    progress: Optional[Callable[[int], None]] = None,
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

    target = (
        Path(output_path).resolve()
        if output_path is not None
        else (
            Path.cwd() / f"submission-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.tar.gz"
        )
    )

    def _write_archive(fileobj: Any) -> None:
        with tarfile.open(fileobj=fileobj, mode="w:gz") as archive:
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

    with open(target, "wb") as raw:
        if progress is not None:
            _write_archive(_CountingWriter(raw, progress))
        else:
            _write_archive(raw)
    return target


def _bytes_io(payload: bytes):
    from io import BytesIO

    return BytesIO(payload)


# ---------------------------------------------------------------------------
# Folder-based submission helpers
# ---------------------------------------------------------------------------

_BUILTIN_IGNORES: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".DS_Store",
    "*.tar.gz",
    "*.tgz",
    "*.zip",
    ".whestignore",
    ".gitignore",
    "manifest.json",
)


def _load_ignore_patterns(root: Path) -> list[str]:
    pats = list(_BUILTIN_IGNORES)
    for fname in (".gitignore", ".whestignore"):
        f = root / fname
        if f.is_file():
            pats += [
                ln.strip()
                for ln in f.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")
            ]
    return pats


def _is_ignored(rel: Path, patterns: list[str]) -> bool:
    parts = rel.parts
    name = rel.name
    for pat in patterns:
        p = pat.rstrip("/")
        if any(fnmatch.fnmatch(seg, p) for seg in parts):
            return True
        if fnmatch.fnmatch(name, p) or fnmatch.fnmatch(str(rel), p):
            return True
    return False


def collect_submission_files(root: "str | Path") -> list[Path]:
    """Files to bundle from a submission folder: every file under ``root`` except
    the built-in + ``.gitignore`` + ``.whestignore`` ignore set. Absolute paths,
    sorted; the archive preserves each path relative to ``root``."""
    root = Path(root).resolve()
    patterns = _load_ignore_patterns(root)
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if _is_ignored(rel, patterns):
            continue
        out.append(path)
    return out


def enforce_submission_caps(files: list[Path]) -> None:
    """Raise ValueError if the submission exceeds the file-count or total-size cap."""
    n = len(files)
    if n > limits.MAX_SUBMISSION_FILES:
        raise ValueError(
            f"Submission has {n} files, over the {limits.MAX_SUBMISSION_FILES}-file cap. "
            f"Exclude files you don't need via .whestignore."
        )
    total = sum(f.stat().st_size for f in files)
    if total > limits.MAX_SUBMISSION_BYTES:
        biggest = sorted(files, key=lambda f: f.stat().st_size, reverse=True)[:3]
        hint = ", ".join(f"{b.name} ({b.stat().st_size / 1e6:.1f} MB)" for b in biggest)
        raise ValueError(
            f"Submission is {total / 1e6:.1f} MB, over the "
            f"{limits.MAX_SUBMISSION_BYTES / 1e6:.0f} MB cap. Biggest: {hint}. "
            f"Exclude large files via .whestignore."
        )


def _local_imports(py_path: Path) -> set[str]:
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _reachable_py(root: Path, entry: Path, py_files: "dict[str, Path]") -> set[str]:
    reachable: set[str] = {entry.stem}
    frontier = [entry]
    while frontier:
        cur = frontier.pop()
        for imp in _local_imports(cur):
            if imp in py_files and imp not in reachable:
                reachable.add(imp)
                frontier.append(py_files[imp])
    return reachable


@dataclass
class SubmissionSummary:
    files: "list[Path]"
    file_count: int
    total_bytes: int
    unreachable_py: "list[str]"


def summarize_submission(estimator_path: "str | Path") -> SubmissionSummary:
    """Preview a submission folder: file list, total size, count, and local .py
    files not reachable (by import) from estimator.py (likely-unused — data files
    are never flagged, they're loaded by runtime path strings)."""
    estimator = Path(estimator_path).resolve()
    root = estimator.parent
    files = collect_submission_files(root)
    py_files = {p.stem: p for p in files if p.suffix == ".py"}
    reachable = _reachable_py(root, estimator, py_files)
    unreachable = sorted(
        str(p.relative_to(root)) for p in files if p.suffix == ".py" and p.stem not in reachable
    )
    total = sum(p.stat().st_size for p in files)
    return SubmissionSummary(
        files=files, file_count=len(files), total_bytes=total, unreachable_py=unreachable
    )
