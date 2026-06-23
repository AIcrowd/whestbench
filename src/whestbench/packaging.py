"""Submission packaging helpers for participant estimator artifacts."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import inspect
import json
import platform
import tarfile
import tempfile
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
    root: Path,
    files: "List[Path]",
    packager_version: str = "0.1.0",
) -> Dict[str, Any]:
    manifest_files = [
        {
            "name": str(p.relative_to(root)),
            "sha256": _sha256(p),
        }
        for p in files
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


def resolve_submission(estimator_path: "str | Path") -> "tuple[Path, Path, str]":
    """Map an ``--estimator`` argument to ``(root, entry, mode)``.

    - A **directory** is a folder submission: ``root`` is the directory, ``entry``
      is ``<dir>/estimator.py`` (which must exist), and the whole folder is bundled.
    - A **file** is a single-file submission: ``root`` is the file's parent and only
      that one file is bundled.
    """
    target = Path(estimator_path).resolve()
    if target.is_dir():
        entry = target / "estimator.py"
        if not entry.is_file():
            raise FileNotFoundError(
                f"Folder submission requires an estimator.py in {target}, but none was found."
            )
        return target, entry, "folder"
    if target.is_file():
        # The grader loads the module named "estimator" (manifest entrypoint). A
        # single-file submission is always shipped AS estimator.py — the packager
        # renames it on the way into the archive (see package_submission), so a
        # file named 01_random.py / my_solution.py "just works" instead of being
        # rejected by the grader with a missing-module error (prod bug 312082).
        return target.parent, target, "file"
    raise FileNotFoundError(f"Estimator path not found: {target}")


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
    root, entry, mode = resolve_submission(estimator_path)
    # Resolve and validate the class entrypoint before packing: the module must
    # import, the Estimator class must resolve, and predict() must accept the
    # (mlp, budget) the grader calls it with. This runs for both `whest package`
    # and `whest submit` (submit packages first), so a broken entrypoint is caught
    # locally instead of failing once per MLP in the grader.
    estimator, metadata = load_estimator_from_path(entry, class_name=class_name)
    _validate_predict_signature(estimator)

    if mode == "file":
        # Single-file submissions are always shipped AS estimator.py: the manifest
        # entrypoint module is hardcoded "estimator", so a file named 01_random.py /
        # my_solution.py would otherwise package locally but be rejected by the grader
        # ("missing module file: estimator.py" — prod bug 312082). Stage a renamed copy
        # in a temp dir and package from there so the archive + manifest agree.
        with tempfile.TemporaryDirectory() as tmp:
            staged_root = Path(tmp)
            staged_entry = staged_root / "estimator.py"
            staged_entry.write_bytes(entry.read_bytes())
            return _finalize_package(
                root=staged_root,
                bundled=[staged_entry],
                class_name=metadata.class_name,
                output_path=output_path,
                progress=progress,
            )

    # Folder mode bundles the whole directory (minus ignores + secrets).
    bundled = collect_submission_files(root)
    if entry not in bundled:
        # A .gitignore/.whestignore pattern matched estimator.py — without this guard
        # we'd ship an archive whose manifest declares an entrypoint it doesn't contain.
        raise ValueError(
            f"estimator.py is excluded by a .gitignore/.whestignore pattern in {root}, "
            f"but it must ship. Remove the pattern that matches it."
        )
    return _finalize_package(
        root=root,
        bundled=bundled,
        class_name=metadata.class_name,
        output_path=output_path,
        progress=progress,
    )


def _validate_predict_signature(estimator: Any) -> None:
    """Reject an estimator whose ``predict`` can't accept the grader's call.

    The grader invokes ``estimator.predict(mlp, budget)``. ``estimator.predict``
    here is a *bound* method, so its signature excludes ``self`` and we bind two
    positional placeholders. Lenient by design: ``*args`` or extra defaulted
    params are fine; only a genuinely incompatible arity (e.g. ``def predict(self)``)
    is rejected. Uninspectable callables (C-implemented) are not blocked.
    """
    try:
        sig = inspect.signature(estimator.predict)
    except (TypeError, ValueError):
        return
    try:
        sig.bind(None, None)
    except TypeError as e:
        raise ValueError(
            f"Estimator.predict has an incompatible signature: the grader calls "
            f"predict(mlp, budget), but binding two positional arguments failed ({e}). "
            f"Expected `def predict(self, mlp, budget)`."
        ) from e


def _finalize_package(
    *,
    root: Path,
    bundled: "List[Path]",
    class_name: str,
    output_path: "Any",
    progress: Optional[Callable[[int], None]],
) -> Path:
    """Build the manifest + deterministic tarball for an already-resolved bundle.

    Shared by single-file (staged estimator.py) and folder packaging so the
    manifest/archive invariants live in exactly one place.
    """
    enforce_submission_caps(bundled)
    manifest = build_manifest(class_name=class_name, root=root, files=bundled)

    # Belt-and-suspenders: the archive we are about to write MUST contain the
    # entrypoint module file the manifest declares (estimator.py). This is the
    # exact consistency that broke in prod bug 312082 — guard it at the source so
    # a manifest can never name an entry file the bundle does not ship.
    arcnames = {str(p.relative_to(root)) for p in bundled}
    entry_file = f"{manifest['entrypoint']['module']}.py"
    if entry_file not in arcnames:
        raise ValueError(
            f"Refusing to package: manifest declares entrypoint module "
            f"{manifest['entrypoint']['module']!r} (the grader imports {entry_file}), but "
            f"{entry_file} is not in the bundle ({sorted(arcnames)})."
        )

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
            for path in bundled:
                archive.add(path, arcname=str(path.relative_to(root)))
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

# Credential/secret patterns. These are ALWAYS excluded — they are never allowed
# in a submission for security reasons (a submission ships to a public leaderboard,
# so a leaked .env / private key would be exposed). Unlike .whestignore patterns,
# these cannot be overridden; rename or remove the file if a match is a false
# positive. ``find_secret_files`` surfaces matches so the CLI can tell the user
# exactly what was dropped and why.
_SECRET_IGNORES: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    ".pypirc",
    ".npmrc",
    ".aws",
    ".ssh",
    ".gnupg",
    "credentials",
    "credentials.*",
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


def _is_secret(rel: Path) -> bool:
    """True if ``rel`` matches a credential/secret pattern. Matched
    case-INSENSITIVELY (fnmatch is case-sensitive on POSIX, but a private key is
    a private key whether it's ``deploy.pem`` or ``deploy.PEM``) against each path
    segment and the basename, so secrets nested in subdirs are caught too."""
    parts = [seg.lower() for seg in rel.parts]
    name = rel.name.lower()
    for pat in _SECRET_IGNORES:
        p = pat.lower().rstrip("/")
        if any(fnmatch.fnmatch(seg, p) for seg in parts) or fnmatch.fnmatch(name, p):
            return True
    return False


def collect_submission_files(root: "str | Path") -> list[Path]:
    """Files to bundle from a submission folder: every regular file under ``root``
    except the built-in + ``.gitignore`` + ``.whestignore`` ignore set and any
    credential/secret file (always excluded for security). Symlinks are skipped —
    they would bake an absolute, possibly out-of-root target into the archive and
    desync the manifest hash. Absolute paths, sorted; the archive preserves each
    path relative to ``root``."""
    root = Path(root).resolve()
    patterns = _load_ignore_patterns(root)
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root)
        if _is_ignored(rel, patterns) or _is_secret(rel):
            continue
        out.append(path)
    return out


def find_secret_files(root: "str | Path") -> list[Path]:
    """Files under ``root`` that were dropped for security (credential/secret
    patterns in :data:`_SECRET_IGNORES`). These are never bundled; this surfaces
    them so the CLI can report exactly what was excluded and why. Absolute paths,
    sorted."""
    root = Path(root).resolve()
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if _is_secret(path.relative_to(root)):
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
    mode: str
    files: "list[Path]"
    file_count: int
    total_bytes: int
    unreachable_py: "list[str]"


def summarize_submission(estimator_path: "str | Path") -> SubmissionSummary:
    """Preview a submission: file list, total size, count, mode, and (folder mode
    only) local .py files not reachable by import from estimator.py (likely-unused;
    data files are never flagged, they're loaded by runtime path strings).

    A directory argument summarizes the whole folder; a file argument summarizes
    just that single file."""
    root, entry, mode = resolve_submission(estimator_path)
    if mode == "folder":
        files = collect_submission_files(root)
        py_files = {p.stem: p for p in files if p.suffix == ".py"}
        reachable = _reachable_py(root, entry, py_files)
        unreachable = sorted(
            str(p.relative_to(root)) for p in files if p.suffix == ".py" and p.stem not in reachable
        )
    else:
        files = [entry]
        unreachable = []
    total = sum(p.stat().st_size for p in files)
    return SubmissionSummary(
        mode=mode,
        files=files,
        file_count=len(files),
        total_bytes=total,
        unreachable_py=unreachable,
    )
