"""The deprecated dataset aliases (push/pull/inspect) have been removed.

`push`/`pull`/`inspect` previously aliased `upload`/`download`/`info` and emitted
a deprecation warning. They are now gone: invoking a removed verb must fail with
argparse's invalid-choice error (exit code 2), and the canonical verbs must stay
registered.
"""

from __future__ import annotations

import pytest

import whestbench.cli as cli


@pytest.mark.parametrize(
    "removed, canonical",
    [("push", "upload"), ("pull", "download"), ("inspect", "info")],
)
def test_removed_dataset_alias_errors(
    removed: str, canonical: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`whest dataset <removed>` exits 2; the error names it and lists canonicals."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["dataset", removed, "x", "--repo", "a/b"])
    assert exc.value.code == 2

    err = capsys.readouterr().err
    assert "invalid choice" in err
    assert removed in err
    # argparse lists the valid choices, which include the canonical verb.
    assert canonical in err


@pytest.mark.parametrize(
    "canonical",
    ["upload", "download", "info", "bake", "merge", "combine-splits", "prepare-arrow"],
)
def test_canonical_dataset_verb_is_registered(canonical: str) -> None:
    """Each canonical verb still parses (`--help` exits 0)."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["dataset", canonical, "--help"])
    assert exc.value.code == 0
