from __future__ import annotations

import re

from whestbench.presentation.models import (
    CommandPresentation,
    ErrorSection,
    KeyValueRow,
    KeyValueSection,
    StepItem,
    StepsSection,
    TableSection,
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


def test_renderers_include_purpose_and_command_for_structured_steps() -> None:
    doc = CommandPresentation(
        command="smoke-test",
        status="success",
        title="WhestBench Report",
        sections=[
            StepsSection(
                title="Next Steps",
                steps=[
                    StepItem(
                        purpose="Create starter files you can edit.",
                        command="whest init ./my-estimator",
                    ),
                    StepItem(
                        purpose="Validate an Estimator implementation.",
                        command="whest validate --estimator ./my-estimator/estimator.py",
                    ),
                ],
            )
        ],
    )

    plain = render_plain_presentation(doc)
    rich = _strip_ansi(render_rich_presentation(doc))

    for text in (
        "Create starter files you can edit.",
        "whest init ./my-estimator",
        "Validate an Estimator implementation.",
        "whest validate --estimator ./my-estimator/estimator.py",
    ):
        assert text in plain
        assert text in rich


def test_renderers_curate_known_error_details_and_keep_unknown_ones() -> None:
    doc = CommandPresentation(
        command="validate",
        status="error",
        title="Error [validate:ESTIMATOR_BAD_SHAPE]",
        sections=[
            ErrorSection(
                title="Failure",
                code="ESTIMATOR_BAD_SHAPE",
                message="Predictions must have shape (2, 4), got (4, 2).",
                details={
                    "expected_shape": [2, 4],
                    "got_shape": [4, 2],
                    "hint": "Returned predictions appear to be transposed.",
                    "cause_hints": ["Predictions must be a 2D array with shape (depth, width)."],
                    "extra_note": "Read the estimator contract.",
                },
            )
        ],
        epilogue_messages=["Use --debug to include a traceback."],
    )

    plain = render_plain_presentation(doc)
    rich = _strip_ansi(render_rich_presentation(doc))

    for text in (
        "Predictions must have shape (2, 4), got (4, 2).",
        "Expected shape: [2, 4]",
        "Got shape: [4, 2]",
        "Hint: Returned predictions appear to be transposed.",
        "Possible causes:",
        "Predictions must be a 2D array with shape (depth, width).",
        "extra_note: Read the estimator contract.",
        "Use --debug to include a traceback.",
    ):
        assert text in plain
        assert text in rich


def test_renderers_do_not_crash_on_array_like_error_detail_values() -> None:
    class _ArrayLikeDetail:
        def __eq__(self, _other: object) -> bool:
            raise ValueError("ambiguous truth value")

        def __str__(self) -> str:
            return "array([1, 2])"

    doc = CommandPresentation(
        command="validate",
        status="error",
        title="Error [validate:ESTIMATOR_BAD_SHAPE]",
        sections=[
            ErrorSection(
                title="Failure",
                code="ESTIMATOR_BAD_SHAPE",
                message="bad shape",
                details={"got_shape": _ArrayLikeDetail()},
            )
        ],
    )

    plain = render_plain_presentation(doc)
    rich = _strip_ansi(render_rich_presentation(doc))

    assert "Got shape: array([1, 2])" in plain
    assert "Got shape: array([1, 2])" in rich


def test_renderers_include_table_section_cells() -> None:
    doc = CommandPresentation(
        command="profile-simulation",
        status="success",
        title="Simulation Profile",
        sections=[
            TableSection(
                title="Detail",
                columns=["Backend", "Dims", "run_mlp", "sample_layer_statistics"],
                rows=[["whest", "256×4×10k", "0.0444s", "0.1135s"]],
            )
        ],
    )

    plain = render_plain_presentation(doc)
    rich = _strip_ansi(render_rich_presentation(doc))

    for text in (
        "Detail",
        "Backend",
        "Dims",
        "whest",
        "256×4×10k",
        "0.0444s",
        "0.1135s",
    ):
        assert text in plain
        assert text in rich


def test_rich_table_renderer_preserves_literal_markup_text() -> None:
    doc = CommandPresentation(
        command="profile-simulation",
        status="success",
        title="Simulation Profile",
        sections=[
            TableSection(
                title="Detail [literal]",
                columns=["Backend", "[dims]"],
                rows=[["[bold]whest[/bold]", "64x4x1k"]],
            )
        ],
    )

    plain = render_plain_presentation(doc)
    rich = _strip_ansi(render_rich_presentation(doc))

    for text in (
        "Detail [literal]",
        "[dims]",
        "[bold]whest[/bold]",
        "64x4x1k",
    ):
        assert text in plain
        assert text in rich
