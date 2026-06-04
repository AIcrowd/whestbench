#!/usr/bin/env python
"""Materialize whest-starterkit participant docs from the pinned SHA into the
Fumadocs content tree. Build-time only; output is gitignored (never committed)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
WEBSITE = ROOT / "website"
LOCK = WEBSITE / "starterkit.lock.json"
DEST = WEBSITE / "content" / "docs" / "participant-guide"
SECTION_DIRS = ["getting-started", "concepts", "how-to", "troubleshooting", "advanced"]
NAMESPACE = "/docs/participant-guide"
RAW_BASE = "https://raw.githubusercontent.com/AIcrowd/whest-starterkit"

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_IMG_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")


def load_pin() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def _title(md: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def _rewrite_images(md: str, rel_slug: str, sha: str) -> str:
    """Point local image references at the upstream raw asset for the pinned SHA.

    Assets are not federated into the content tree, and Fumadocs/Next resolves a
    local image target as a static module import (a missing-module build error).
    Resolving to an absolute raw URL renders the image and skips that import. Runs
    before ``_rewrite_links`` so the resulting ``https://`` target is left alone.
    """
    src_dir = PurePosixPath("docs") / PurePosixPath(rel_slug).parent

    def repl(m: re.Match) -> str:
        prefix, target, suffix = m.group(1), m.group(2), m.group(3)
        if target.startswith(("http://", "https://", "//", "data:")):
            return m.group(0)
        resolved = os.path.normpath((src_dir / target).as_posix())
        return f"{prefix}{RAW_BASE}/{sha}/{resolved}{suffix}"

    return _IMG_RE.sub(repl, md)


def _rewrite_links(md: str) -> str:
    def repl(m: re.Match) -> str:
        target = m.group(1)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        target = re.sub(r"\.mdx?($|#)", r"\1", target)
        cleaned = target.lstrip("./")
        return f"]({NAMESPACE}/{cleaned})"

    return re.sub(r"\]\(([^)]+)\)", repl, md)


def _escape_inline(text: str) -> str:
    parts = re.split(r"(`[^`]*`)", text)
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("<", "&lt;").replace("{", "\\{")
    return "".join(parts)


def _sanitize_mdx(md: str) -> str:
    out: list[str] = []
    in_fence = False
    for line in md.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            out.append(_escape_inline(line))
    return "\n".join(out)


def to_mdx(md: str, rel_slug: str, sha: str) -> str:
    title = _title(md)
    body = _sanitize_mdx(_rewrite_links(_rewrite_images(md, rel_slug, sha)))
    banner = (
        f"> Sourced from [whest-starterkit](https://github.com/AIcrowd/whest-starterkit/"
        f"blob/{sha}/docs/{rel_slug}.md) @ `{sha[:7]}`.\n\n"
    )
    fm = f"---\ntitle: {json.dumps(title)}\n---\n\n"
    return fm + banner + body


def fetch(sha: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="whest-starterkit-"))
    subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
    subprocess.run(
        ["git", "fetch", "-q", "--depth", "1", "https://github.com/AIcrowd/whest-starterkit", sha],
        cwd=tmp,
        check=True,
    )
    subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=tmp, check=True)
    return tmp


def materialize() -> list[Path]:
    pin = load_pin()
    sha = pin["sha"]
    src = fetch(sha)
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    written: list[Path] = []
    pages: list[str] = []
    for section in SECTION_DIRS:
        sdir = src / "docs" / section
        if not sdir.is_dir():
            continue
        (DEST / section).mkdir(parents=True, exist_ok=True)
        for md_path in sorted(sdir.glob("*.md")):
            stem = md_path.stem.lower()
            rel_slug = f"{section}/{stem}"
            out = DEST / section / f"{stem}.mdx"
            out.write_text(
                to_mdx(md_path.read_text(encoding="utf-8"), rel_slug, sha),
                encoding="utf-8",
            )
            written.append(out)
        pages.append(section)
    (DEST / "meta.json").write_text(
        json.dumps({"title": "Participant Guide", "pages": pages}, indent=2) + "\n",
        encoding="utf-8",
    )
    index = (
        "---\ntitle: Participant Guide\n"
        "description: Tutorial and how-to for the challenge, federated from whest-starterkit.\n---\n\n"
        "# Participant Guide\n\nFederated from "
        f"[`AIcrowd/whest-starterkit`](https://github.com/AIcrowd/whest-starterkit/tree/{sha}) "
        f"@ `{sha[:7]}`.\n"
    )
    (DEST / "index.mdx").write_text(index, encoding="utf-8")
    shutil.rmtree(src, ignore_errors=True)
    return written


if __name__ == "__main__":
    files = materialize()
    print(f"Materialized {len(files)} participant-guide pages")
