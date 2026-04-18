from __future__ import annotations

import re

from whestbench.presentation.models import (
    CommandPresentation,
    KeyValueRow,
    KeyValueSection,
    StepsSection,
)
from whestbench.presentation.render_plain import render_plain_presentation
from whestbench.presentation.render_rich import render_rich_presentation


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_renderers_preserve_same_key_facts() -> None:
    doc = CommandPresentation(
        command="smoke-test [nightly]",
        status="success",
        title="WhestBench Report [beta]",
        subtitle="settled result [ok]",
        sections=[
            KeyValueSection(
                title="Run Context [system]",
                rows=[
                    KeyValueRow(label="MLPs [count]", value="3"),
                    KeyValueRow(label="Width", value="100"),
                ],
            ),
            StepsSection(
                title="Next Steps [follow-up]",
                steps=[
                    "whest init ./my-estimator [template]",
                    "whest validate --estimator ./my-estimator/estimator.py",
                ],
            ),
        ],
        epilogue_messages=[
            "Use --json [machine-readable] to preserve the full structured report when integrating with automated systems."
        ],
    )

    plain = render_plain_presentation(doc)
    rich = _strip_ansi(render_rich_presentation(doc))

    for needle in (
        "WhestBench Report [beta]",
        "Command: smoke-test [nightly]",
        "Status: success",
        "Run Context [system]",
        "MLPs [count]",
        "3",
        "Next Steps [follow-up]",
        "whest init ./my-estimator [template]",
    ):
        assert needle in plain
        assert needle in rich

    for fragment in (
        "Use --json [machine-readable]",
        "preserve the full structured report",
        "integrating with automated systems.",
    ):
        assert fragment in rich
