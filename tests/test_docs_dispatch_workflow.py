from __future__ import annotations

from pathlib import Path


def test_docs_dispatch_workflow_triggers_only_for_main_docs_changes() -> None:
    source = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "docs-dispatch.yml"
    ).read_text()

    assert "branches: [main]" in source
    assert "docs/**" in source
    assert "docs-kit/**" in source
    assert "docs/unified-docs-process.md" in source


def test_docs_dispatch_workflow_targets_flopscope_docs_repository_dispatch() -> None:
    source = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "docs-dispatch.yml"
    ).read_text()

    assert "repos/AIcrowd/flopscope-docs/dispatches" in source
    assert '"event_type": "source-updated"' in source
    assert "AICROWD_DOCS_DISPATCH_TOKEN" in source
