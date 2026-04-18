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
        command="smoke-test",
        status="success",
        title="WhestBench Report",
        subtitle="settled result",
        sections=[
            KeyValueSection(
                title="Run Context",
                rows=[
                    KeyValueRow(label="MLPs", value="3"),
                    KeyValueRow(label="Width", value="100"),
                ],
            ),
            StepsSection(
                title="Next Steps",
                steps=[
                    "whest init ./my-estimator",
                    "whest validate --estimator ./my-estimator/estimator.py",
                ],
            ),
        ],
        epilogue_messages=["Use --json for JSON output when calling from automated agents."],
    )

    plain = render_plain_presentation(doc)
    rich = _strip_ansi(render_rich_presentation(doc))

    for needle in (
        "WhestBench Report",
        "Run Context",
        "MLPs",
        "3",
        "Next Steps",
        "whest init ./my-estimator",
        "Use --json for JSON output when calling from automated agents.",
    ):
        assert needle in plain
        assert needle in rich
