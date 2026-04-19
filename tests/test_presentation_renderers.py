from __future__ import annotations

import re

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whestbench.presentation.blocks import build_budget_breakdown_block, build_score_block
from whestbench.presentation.human import render_document
from whestbench.presentation.models import (
    BudgetBreakdownGauge,
    BudgetBreakdownNamespaceRow,
    BudgetBreakdownOverBudgetRow,
    BudgetBreakdownSection,
    CommandPresentation,
    ErrorSection,
    KeyValueRow,
    KeyValueSection,
    StepItem,
    StepsSection,
    TableSection,
)
from whestbench.presentation.render_plain import render_plain_presentation
from whestbench.presentation.render_rich import _section_renderables, render_rich_presentation


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


def test_renderers_render_budget_breakdowns_before_final_score() -> None:
    doc = CommandPresentation(
        command="run",
        status="success",
        title="WhestBench Report",
        sections=[
            BudgetBreakdownSection(
                title="Sampling Budget Breakdown (Ground Truth)",
                available=True,
                total_flops="80",
                tracked_time="0.020000s",
                untracked_time="0.010000s",
                namespace_rows=[
                    BudgetBreakdownNamespaceRow(
                        namespace="sampling.sample_layer_statistics",
                        total_flops="80",
                        percent_of_section_flops="100.0%",
                        mean_flops_per_mlp="40",
                        tracked_time="0.020000s",
                    )
                ],
                source_note="restored from dataset metadata for the MLPs used in this run.",
            ),
            BudgetBreakdownSection(
                title="Estimator Budget Breakdown",
                available=True,
                total_flops="90",
                tracked_time="0.030000s",
                untracked_time="0.010000s",
                namespace_rows=[
                    BudgetBreakdownNamespaceRow(
                        namespace="estimator.estimator-client",
                        total_flops="90",
                        percent_of_section_flops="100.0%",
                        mean_flops_per_mlp="45",
                        tracked_time="0.030000s",
                    )
                ],
                gauge=BudgetBreakdownGauge(
                    label="Estimator FLOPs",
                    bar="[##########----------]",
                    overflow=False,
                    percent_of_budget="45%",
                    budget_label="100",
                    worst_mlp_percent="60%",
                ),
                over_budget_rows=[
                    BudgetBreakdownOverBudgetRow(
                        mlp_index=1,
                        flops_used="120",
                        percent_of_budget="120%",
                    )
                ],
                over_budget_summary="1 of 2 MLPs exceeded the per-MLP FLOP cap",
            ),
            KeyValueSection(
                title="Final Score",
                rows=[
                    KeyValueRow(label="Primary Score", value="0.123"),
                    KeyValueRow(label="Secondary Score", value="0.456"),
                ],
            ),
        ],
    )

    plain = render_plain_presentation(doc)
    rich = _strip_ansi(render_rich_presentation(doc))

    for rendered in (plain, rich):
        assert rendered.index("Sampling Budget Breakdown (Ground Truth)") < rendered.index(
            "Estimator Budget Breakdown"
        )
        assert rendered.index("Estimator Budget Breakdown") < rendered.index("Final Score")
        assert "restored from dataset metadata for the MLPs used in this run." in rendered
        assert "aggregated across all evaluated MLPs" in rendered
        assert "Estimator FLOPs" not in rendered
        assert "1 of 2 MLPs exceeded the per-MLP FLOP cap" not in rendered


def test_renderers_show_unavailable_budget_breakdown_message() -> None:
    doc = CommandPresentation(
        command="run",
        status="success",
        title="WhestBench Report",
        sections=[
            BudgetBreakdownSection(
                title="Sampling Budget Breakdown (Ground Truth)",
                available=False,
                unavailable_message=(
                    "Ground-truth sampling baseline is unavailable for this dataset. "
                    "Recreate the dataset with a newer whestbench to compare against sampling."
                ),
            )
        ],
    )

    plain = render_plain_presentation(doc)
    rich = _strip_ansi(render_rich_presentation(doc))

    for rendered in (plain, rich):
        assert "Sampling Budget Breakdown (Ground Truth)" in rendered
        assert "Ground-truth sampling baseline is unavailable for this dataset." in rendered
        assert "newer whestbench" in rendered


def test_renderers_match_main_style_run_score_and_breakdown_information() -> None:
    doc = CommandPresentation(
        command="run",
        status="success",
        title="WhestBench Report",
        sections=[
            BudgetBreakdownSection(
                title="Sampling Budget Breakdown (Ground Truth)",
                available=True,
                total_flops="1.33e+06",
                tracked_time="0.000841s",
                untracked_time="0.003194s",
                namespace_rows=[
                    BudgetBreakdownNamespaceRow(
                        namespace="sampling.sample_layer_statistics",
                        total_flops="1.33e+06",
                        percent_of_section_flops="100.0%",
                        mean_flops_per_mlp="1.33e+06",
                        tracked_time="0.000841s",
                    )
                ],
            ),
            BudgetBreakdownSection(
                title="Estimator Budget Breakdown",
                available=True,
                total_flops="4.84e+07",
                tracked_time="0.005277s",
                untracked_time="0.012066s",
                namespace_rows=[
                    BudgetBreakdownNamespaceRow(
                        namespace="estimator.estimator-client",
                        total_flops="4.84e+07",
                        percent_of_section_flops="100.0%",
                        mean_flops_per_mlp="4.84e+07",
                        tracked_time="0.005277s",
                    )
                ],
                gauge=BudgetBreakdownGauge(
                    label="Estimator FLOPs",
                    bar="[##########----------]",
                    overflow=False,
                    percent_of_budget="48%",
                    budget_label="1.00e+08",
                ),
            ),
            TableSection(
                title="Final Score",
                columns=["metric", "value"],
                rows=[
                    ["Primary Score [primary_score]", "0.01329615"],
                    ["Secondary Score [secondary_score]", "0.03770037"],
                    ["Best MLP Score [best_mlp_score]", "0.01329615"],
                    ["Worst MLP Score [worst_mlp_score]", "0.01329615"],
                ],
                subtitle="lower MSE is better; primary score = mean across MLPs of final-layer MSE",
            ),
        ],
    )

    plain = render_plain_presentation(doc)
    rich = _strip_ansi(render_rich_presentation(doc))

    for rendered in (plain, rich):
        assert rendered.index("Sampling Budget Breakdown (Ground Truth)") < rendered.index(
            "Estimator Budget Breakdown"
        )
        assert rendered.index("Estimator Budget Breakdown") < rendered.index("Final Score")
        assert "Total FLOPs [flops_used]" in rendered
        assert "Tracked Time [tracked_time_s]" in rendered
        assert "Untracked Time [untracked_time_s]" in rendered
        assert "aggregated across all evaluated MLPs" in rendered
        assert "Primary Score [primary_score]" in rendered
        assert "Secondary Score [secondary_score]" in rendered
        assert "Best MLP Score [best_mlp_score]" in rendered
        assert "Worst MLP Score [worst_mlp_score]" in rendered
        assert "lower MSE is better" in rendered
        assert "primary score = mean across MLPs" in rendered
        assert "Estimator FLOPs" not in rendered


def test_budget_breakdown_rich_renderer_uses_centered_rich_tables() -> None:
    section = BudgetBreakdownSection(
        title="Sampling Budget Breakdown (Ground Truth)",
        available=True,
        total_flops="1.33e+06",
        tracked_time="0.000841s",
        untracked_time="0.003194s",
        namespace_rows=[
            BudgetBreakdownNamespaceRow(
                namespace="sampling.sample_layer_statistics",
                total_flops="1.33e+06",
                percent_of_section_flops="100.0%",
                mean_flops_per_mlp="1.33e+06",
                tracked_time="0.000841s",
            )
        ],
    )

    renderables = _section_renderables(section)

    assert len(renderables) == 1
    panel = renderables[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Group)
    children = list(panel.renderable.renderables)
    assert isinstance(children[0], Align)
    assert isinstance(children[0].renderable, Table)
    assert isinstance(children[1], Align)
    assert isinstance(children[1].renderable, Table)


def test_budget_breakdown_rich_summary_labels_keep_old_color_spans() -> None:
    section = BudgetBreakdownSection(
        title="Sampling Budget Breakdown (Ground Truth)",
        available=True,
        total_flops="1.33e+06",
        tracked_time="0.000841s",
        untracked_time="0.003194s",
    )

    panel = _section_renderables(section)[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Group)
    summary = list(panel.renderable.renderables)[0]
    assert isinstance(summary, Align)
    assert isinstance(summary.renderable, Table)

    label_cells = summary.renderable.columns[0]._cells
    first_label = label_cells[0]
    second_label = label_cells[1]
    third_label = label_cells[2]

    assert isinstance(first_label, Text)
    assert first_label.plain == "Total FLOPs [flops_used]"
    assert [str(span.style) for span in first_label.spans] == [
        "bold bright_yellow",
        "bold bright_white",
    ]
    assert isinstance(second_label, Text)
    assert second_label.plain == "Tracked Time [tracked_time_s]"
    assert [str(span.style) for span in second_label.spans] == [
        "bold bright_green",
        "bold bright_white",
    ]
    assert isinstance(third_label, Text)
    assert third_label.plain == "Untracked Time [untracked_time_s]"
    assert [str(span.style) for span in third_label.spans] == [
        "bold bright_green",
        "bold bright_white",
    ]


def test_final_score_rich_renderer_keeps_old_color_coding() -> None:
    section = TableSection(
        title="Final Score",
        columns=["metric", "value"],
        rows=[
            ["Primary Score [primary_score]", "0.01329615"],
            ["Secondary Score [secondary_score]", "0.03770037"],
            ["Best MLP Score [best_mlp_score]", "0.01329615"],
            ["Worst MLP Score [worst_mlp_score]", "0.01329615"],
        ],
        subtitle="lower MSE is better; primary score = mean across MLPs of final-layer MSE",
        align_center=True,
        border_style="bright_cyan",
    )

    panel = _section_renderables(section)[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Align)
    assert isinstance(panel.renderable.renderable, Table)
    table = panel.renderable.renderable

    metric_cells = table.columns[0]._cells
    value_cells = table.columns[1]._cells

    assert isinstance(metric_cells[0], Text)
    assert metric_cells[0].plain == "Primary Score [primary_score]"
    assert [str(span.style) for span in metric_cells[0].spans] == [
        "bold bright_green",
        "bold bright_white",
    ]
    assert isinstance(metric_cells[1], Text)
    assert metric_cells[1].plain == "Secondary Score [secondary_score]"
    assert [str(span.style) for span in metric_cells[1].spans] == [
        "bold bright_cyan",
        "bold bright_white",
    ]
    assert isinstance(metric_cells[2], Text)
    assert metric_cells[2].plain == "Best MLP Score [best_mlp_score]"
    assert [str(span.style) for span in metric_cells[2].spans] == [
        "bold green",
        "bold bright_white",
    ]
    assert isinstance(metric_cells[3], Text)
    assert metric_cells[3].plain == "Worst MLP Score [worst_mlp_score]"
    assert [str(span.style) for span in metric_cells[3].spans] == [
        "bold yellow",
        "bold bright_white",
    ]

    assert value_cells[0] == "[bold bright_green]0.01329615[/]"
    assert value_cells[1] == "[cyan]0.03770037[/]"
    assert value_cells[2] == "[green]0.01329615[/]"
    assert value_cells[3] == "[yellow]0.01329615[/]"


def test_shared_human_document_renders_budget_before_final_score_in_rich_and_plain() -> None:
    budget_section = BudgetBreakdownSection(
        title="Estimator Budget Breakdown",
        available=True,
        total_flops="4.84e+07",
        tracked_time="0.005277s",
        untracked_time="0.012066s",
        namespace_rows=[
            BudgetBreakdownNamespaceRow(
                namespace="estimator.estimator-client",
                total_flops="4.84e+07",
                percent_of_section_flops="100.0%",
                mean_flops_per_mlp="4.84e+07",
                tracked_time="0.005277s",
            )
        ],
    )
    score_section = TableSection(
        title="Final Score",
        columns=["metric", "value"],
        rows=[
            ["Primary Score [primary_score]", "0.01329615"],
            ["Secondary Score [secondary_score]", "0.03770037"],
        ],
        subtitle="lower MSE is better; primary score = mean across MLPs of final-layer MSE",
    )

    blocks = [
        build_budget_breakdown_block(budget_section),
        build_score_block(score_section),
    ]

    rich = _strip_ansi(
        render_document(title="WhestBench Report", blocks=blocks, output_format="rich")
    )
    plain = render_document(title="WhestBench Report", blocks=blocks, output_format="plain")

    for rendered in (rich, plain):
        assert rendered.index("Estimator Budget Breakdown") < rendered.index("Final Score")
        assert "Total FLOPs [flops_used]" in rendered
        assert "Tracked Time [tracked_time_s]" in rendered
        assert "Untracked Time [untracked_time_s]" in rendered
        assert "Primary Score [primary_score]" in rendered
        assert "Secondary Score [secondary_score]" in rendered
        assert "lower MSE is better" in rendered
