from __future__ import annotations

import re

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whestbench.presentation.blocks import (
    build_budget_breakdown_block,
    build_score_block,
    build_section_renderables,
)
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
from whestbench.presentation.presenters import render_command_presentation


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _render_plain(doc: CommandPresentation) -> str:
    return render_command_presentation(
        doc,
        output_format="plain",
        force_terminal=False,
        width=200,
    )


def _render_rich(doc: CommandPresentation) -> str:
    return render_command_presentation(
        doc,
        output_format="rich",
        force_terminal=True,
        width=200,
    )


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

    plain = _render_plain(doc)
    rich = _strip_ansi(_render_rich(doc))

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

    plain = _render_plain(doc)
    rich = _strip_ansi(_render_rich(doc))

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

    plain = _render_plain(doc)
    rich = _strip_ansi(_render_rich(doc))

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

    plain = _render_plain(doc)
    rich = _strip_ansi(_render_rich(doc))

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
                rows=[["flopscope", "256×4×10k", "0.0444s", "0.1135s"]],
            )
        ],
    )

    plain = _render_plain(doc)
    rich = _strip_ansi(_render_rich(doc))

    for text in (
        "Detail",
        "Backend",
        "Dims",
        "flopscope",
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
                rows=[["[bold]flopscope[/bold]", "64x4x1k"]],
            )
        ],
    )

    plain = _render_plain(doc)
    rich = _strip_ansi(_render_rich(doc))

    for text in (
        "Detail [literal]",
        "[dims]",
        "[bold]flopscope[/bold]",
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
                flopscope_backend_time="0.020000s",
                flopscope_overhead_time="0.005000s",
                residual_wall_time="0.010000s",
                namespace_rows=[
                    BudgetBreakdownNamespaceRow(
                        namespace="sampling.sample_layer_statistics",
                        total_flops="80",
                        percent_of_section_flops="100.0%",
                        mean_flops_per_mlp="40",
                        flopscope_backend_time="0.020000s",
                        flopscope_overhead_time="0.005000s",
                    )
                ],
                source_note="restored from dataset metadata for the MLPs used in this run.",
            ),
            BudgetBreakdownSection(
                title="Estimator Budget Breakdown",
                available=True,
                total_flops="90",
                flopscope_backend_time="0.030000s",
                flopscope_overhead_time="0.007500s",
                residual_wall_time="0.010000s",
                namespace_rows=[
                    BudgetBreakdownNamespaceRow(
                        namespace="estimator.estimator-client",
                        total_flops="90",
                        percent_of_section_flops="100.0%",
                        mean_flops_per_mlp="45",
                        flopscope_backend_time="0.030000s",
                        flopscope_overhead_time="0.007500s",
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
                        reason="BUDGET",
                        metric_name="C_m",
                        metric_value="120",
                        flops_used="120",
                        percent_of_budget="120%",
                    )
                ],
                over_budget_summary="1 of 2 MLPs failed (0 combined, 1 FLOP, 0 residual, 0 time, 0 error). All counted as failures.",
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

    plain = _render_plain(doc)
    rich = _strip_ansi(_render_rich(doc))

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

    plain = _render_plain(doc)
    rich = _strip_ansi(_render_rich(doc))

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
                flopscope_backend_time="0.000841s",
                flopscope_overhead_time="0.000210s",
                residual_wall_time="0.003194s",
                namespace_rows=[
                    BudgetBreakdownNamespaceRow(
                        namespace="sampling.sample_layer_statistics",
                        total_flops="1.33e+06",
                        percent_of_section_flops="100.0%",
                        mean_flops_per_mlp="1.33e+06",
                        flopscope_backend_time="0.000841s",
                        flopscope_overhead_time="0.000210s",
                    )
                ],
            ),
            BudgetBreakdownSection(
                title="Estimator Budget Breakdown",
                available=True,
                total_flops="4.84e+07",
                flopscope_backend_time="0.005277s",
                flopscope_overhead_time="0.001319s",
                residual_wall_time="0.012066s",
                namespace_rows=[
                    BudgetBreakdownNamespaceRow(
                        namespace="estimator.estimator-client",
                        total_flops="4.84e+07",
                        percent_of_section_flops="100.0%",
                        mean_flops_per_mlp="4.84e+07",
                        flopscope_backend_time="0.005277s",
                        flopscope_overhead_time="0.001319s",
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
                    [
                        "Adjusted Final-Layer Score [adjusted_final_layer_score]",
                        "0.01329615  ← primary score",
                    ],
                    ["Raw Final-Layer MSE [final_layer_mse]", "0.01200000"],
                    ["All-Layers MSE [all_layers_mse]", "0.03770037"],
                    ["Best MLP [best_mlp_adjusted_final_layer_score]", "0.01329615"],
                    ["Worst MLP [worst_mlp_adjusted_final_layer_score]", "0.01329615"],
                    ["Mean Score Multiplier [mean_score_multiplier]", "0.90000000"],
                    ["Mean Compute Utilization [mean_compute_utilization]", "0.50000000"],
                    ["Failed MLPs [n_failed_mlps]", "0 of 1"],
                ],
                subtitle="lower is better; adjusted_final_layer_score = final_layer_mse × max(0.1, C_m/B_m); failure → × 1.0",
            ),
        ],
    )

    plain = _render_plain(doc)
    rich = _strip_ansi(_render_rich(doc))

    for rendered in (plain, rich):
        assert rendered.index("Sampling Budget Breakdown (Ground Truth)") < rendered.index(
            "Estimator Budget Breakdown"
        )
        assert rendered.index("Estimator Budget Breakdown") < rendered.index("Final Score")
        assert "Total FLOPs [flops_used]" in rendered
        assert "Flopscope Backend [flopscope_backend_time_s]" in rendered
        assert "Flopscope Overhead [flopscope_overhead_time_s]" in rendered
        assert "Residual Wall Time [residual_wall_time_s]" in rendered
        assert "aggregated across all evaluated MLPs" in rendered
        assert "Adjusted Final-Layer Score" in rendered
        assert "adjusted_final_layer_score" in rendered
        assert "All-Layers MSE [all_layers_mse]" in rendered
        assert "best_mlp_adjusted_final_layer_score" in rendered
        assert "worst_mlp_adjusted_final_layer_score" in rendered
        assert "lower is better" in rendered
        assert "max(0.1, C_m/" in rendered
        assert "Estimator FLOPs" not in rendered


def test_budget_breakdown_rich_renderer_uses_centered_rich_tables() -> None:
    section = BudgetBreakdownSection(
        title="Sampling Budget Breakdown (Ground Truth)",
        available=True,
        total_flops="1.33e+06",
        flopscope_backend_time="0.000841s",
        flopscope_overhead_time="0.000210s",
        residual_wall_time="0.003194s",
        namespace_rows=[
            BudgetBreakdownNamespaceRow(
                namespace="sampling.sample_layer_statistics",
                total_flops="1.33e+06",
                percent_of_section_flops="100.0%",
                mean_flops_per_mlp="1.33e+06",
                flopscope_backend_time="0.000841s",
                flopscope_overhead_time="0.000210s",
            )
        ],
    )

    renderables = build_section_renderables(section)

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
        flopscope_backend_time="0.000841s",
        flopscope_overhead_time="0.000210s",
        residual_wall_time="0.003194s",
    )

    panel = build_section_renderables(section)[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Group)
    summary = list(panel.renderable.renderables)[0]
    assert isinstance(summary, Align)
    assert isinstance(summary.renderable, Table)

    label_cells = summary.renderable.columns[0]._cells
    first_label = label_cells[0]
    second_label = label_cells[1]
    third_label = label_cells[2]
    fourth_label = label_cells[3]

    assert isinstance(first_label, Text)
    assert first_label.plain == "Total FLOPs [flops_used]"
    assert [str(span.style) for span in first_label.spans] == [
        "bold bright_yellow",
        "dim",
    ]
    assert isinstance(second_label, Text)
    assert second_label.plain == "Flopscope Backend [flopscope_backend_time_s]"
    assert [str(span.style) for span in second_label.spans] == [
        "bold bright_green",
        "dim",
    ]
    assert isinstance(third_label, Text)
    assert third_label.plain == "Flopscope Overhead [flopscope_overhead_time_s]"
    assert [str(span.style) for span in third_label.spans] == [
        "bold bright_yellow",
        "dim",
    ]
    assert isinstance(fourth_label, Text)
    assert fourth_label.plain == "Residual Wall Time [residual_wall_time_s]"
    assert [str(span.style) for span in fourth_label.spans] == [
        "bold bright_green",
        "dim",
    ]


def test_final_score_rich_renderer_uses_new_color_coding() -> None:
    section = TableSection(
        title="Final Score",
        columns=["metric", "value"],
        rows=[
            [
                "Adjusted Final-Layer Score [adjusted_final_layer_score]",
                "0.01329615  ← primary score",
            ],
            ["Raw Final-Layer MSE [final_layer_mse]", "0.01200000"],
            ["All-Layers MSE [all_layers_mse]", "0.03770037"],
            ["Best MLP [best_mlp_adjusted_final_layer_score]", "0.01329615"],
            ["Worst MLP [worst_mlp_adjusted_final_layer_score]", "0.01329615"],
            ["Mean Score Multiplier [mean_score_multiplier]", "0.90000000"],
            ["Mean Compute Utilization [mean_compute_utilization]", "0.50000000"],
            ["Failed MLPs [n_failed_mlps]", "0 of 3"],
        ],
        subtitle="lower is better; adjusted_final_layer_score = final_layer_mse × max(0.1, C_m/B_m); failure → × 1.0",
        align_center=True,
        border_style="bright_cyan",
    )

    panel = build_section_renderables(section)[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Align)
    assert isinstance(panel.renderable.renderable, Table)
    table = panel.renderable.renderable

    # With dividers, the table has 10 rows: 8 data + 2 divider rows.
    # Data rows are at indices 0, 1, 2, 4, 5, 7, 8, 9 (with dividers at 3 and 6).
    metric_cells = table.columns[0]._cells
    value_cells = table.columns[1]._cells

    assert isinstance(metric_cells[0], Text)
    assert metric_cells[0].plain == "Adjusted Final-Layer Score [adjusted_final_layer_score]"
    assert [str(span.style) for span in metric_cells[0].spans] == [
        "bold bright_green",
        "dim",
    ]
    assert isinstance(metric_cells[1], Text)
    assert metric_cells[1].plain == "Raw Final-Layer MSE [final_layer_mse]"
    assert [str(span.style) for span in metric_cells[1].spans] == [
        "bold cyan",
        "dim",
    ]
    assert isinstance(metric_cells[2], Text)
    assert metric_cells[2].plain == "All-Layers MSE [all_layers_mse]"
    assert [str(span.style) for span in metric_cells[2].spans] == [
        "bold cyan",
        "dim",
    ]
    # Index 3 is a divider row (Text("────────"))
    assert isinstance(metric_cells[4], Text)
    assert metric_cells[4].plain == "Best MLP [best_mlp_adjusted_final_layer_score]"
    assert [str(span.style) for span in metric_cells[4].spans] == [
        "bold green",
        "dim",
    ]
    assert isinstance(metric_cells[5], Text)
    assert metric_cells[5].plain == "Worst MLP [worst_mlp_adjusted_final_layer_score]"
    assert [str(span.style) for span in metric_cells[5].spans] == [
        "bold yellow",
        "dim",
    ]

    assert value_cells[0] == "[bold bright_green]0.01329615  ← primary score[/]"
    assert value_cells[1] == "[cyan]0.01200000[/]"
    assert value_cells[2] == "[cyan]0.03770037[/]"
    assert value_cells[4] == "[green]0.01329615[/]"
    assert value_cells[5] == "[yellow]0.01329615[/]"


def test_shared_human_document_renders_budget_before_final_score_in_rich_and_plain() -> None:
    budget_section = BudgetBreakdownSection(
        title="Estimator Budget Breakdown",
        available=True,
        total_flops="4.84e+07",
        flopscope_backend_time="0.005277s",
        flopscope_overhead_time="0.001319s",
        residual_wall_time="0.012066s",
        namespace_rows=[
            BudgetBreakdownNamespaceRow(
                namespace="estimator.estimator-client",
                total_flops="4.84e+07",
                percent_of_section_flops="100.0%",
                mean_flops_per_mlp="4.84e+07",
                flopscope_backend_time="0.005277s",
                flopscope_overhead_time="0.001319s",
            )
        ],
    )
    score_section = TableSection(
        title="Final Score",
        columns=["metric", "value"],
        rows=[
            [
                "Adjusted Final-Layer Score [adjusted_final_layer_score]",
                "0.01329615  ← primary score",
            ],
            ["Raw Final-Layer MSE [final_layer_mse]", "0.01200000"],
            ["All-Layers MSE [all_layers_mse]", "0.03770037"],
            ["Best MLP [best_mlp_adjusted_final_layer_score]", "0.01329615"],
            ["Worst MLP [worst_mlp_adjusted_final_layer_score]", "0.01329615"],
            ["Mean Score Multiplier [mean_score_multiplier]", "0.90000000"],
            ["Mean Compute Utilization [mean_compute_utilization]", "0.50000000"],
            ["Failed MLPs [n_failed_mlps]", "0 of 1"],
        ],
        subtitle="lower is better; adjusted_final_layer_score = final_layer_mse × max(0.1, C_m/B_m); failure → × 1.0",
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
        assert "Flopscope Backend [flopscope_backend_time_s]" in rendered
        assert "Flopscope Overhead [flopscope_overhead_time_s]" in rendered
        assert "Residual Wall Time [residual_wall_time_s]" in rendered
        assert "Adjusted Final-Layer Score" in rendered
        assert "adjusted_final_layer_score" in rendered
        assert "lower is better" in rendered


def test_shared_human_plain_output_uses_rich_safe_text_layout() -> None:
    budget_section = BudgetBreakdownSection(
        title="Estimator Budget Breakdown",
        available=True,
        total_flops="4.84e+07",
        flopscope_backend_time="0.005277s",
        flopscope_overhead_time="0.001319s",
        residual_wall_time="0.012066s",
        namespace_rows=[
            BudgetBreakdownNamespaceRow(
                namespace="estimator.estimator-client",
                total_flops="4.84e+07",
                percent_of_section_flops="100.0%",
                mean_flops_per_mlp="4.84e+07",
                flopscope_backend_time="0.005277s",
                flopscope_overhead_time="0.001319s",
            )
        ],
    )
    score_section = TableSection(
        title="Final Score",
        columns=["metric", "value"],
        rows=[
            [
                "Adjusted Final-Layer Score [adjusted_final_layer_score]",
                "0.01329615  ← primary score",
            ],
            ["Raw Final-Layer MSE [final_layer_mse]", "0.01200000"],
            ["All-Layers MSE [all_layers_mse]", "0.03770037"],
            ["Best MLP [best_mlp_adjusted_final_layer_score]", "0.01329615"],
            ["Worst MLP [worst_mlp_adjusted_final_layer_score]", "0.01329615"],
            ["Mean Score Multiplier [mean_score_multiplier]", "0.90000000"],
            ["Mean Compute Utilization [mean_compute_utilization]", "0.50000000"],
            ["Failed MLPs [n_failed_mlps]", "0 of 1"],
        ],
        subtitle="lower is better; adjusted_final_layer_score = final_layer_mse × max(0.1, C_m/B_m); failure → × 1.0",
    )

    plain = render_document(
        title="WhestBench Report",
        blocks=[
            build_budget_breakdown_block(budget_section),
            build_score_block(score_section),
        ],
        output_format="plain",
        width=100,
    )

    assert re.search(r"^[╭┌+].*WhestBench Report", plain, re.MULTILINE)
    assert "metric | value" not in plain
    assert "namespace | total flops" not in plain


def test_shared_human_plain_output_keeps_long_values_readable_under_rich_safe_text_rendering() -> (
    None
):
    budget_section = BudgetBreakdownSection(
        title="Estimator Budget Breakdown",
        available=True,
        total_flops="123456789012345678901234567890",
        flopscope_backend_time="0.12345678901234567890s",
        flopscope_overhead_time="0.030864s",
        residual_wall_time="0.98765432109876543210s",
        namespace_rows=[
            BudgetBreakdownNamespaceRow(
                namespace="sampling.sample_layer_statistics.really_long_namespace",
                total_flops="123456789012345678901234567890",
                percent_of_section_flops="100.0000000000%",
                mean_flops_per_mlp="12345678901234567890",
                flopscope_backend_time="0.12345678901234567890s",
                flopscope_overhead_time="0.030864s",
            )
        ],
    )
    score_section = TableSection(
        title="Final Score",
        columns=["metric", "value"],
        rows=[
            [
                "Adjusted Final-Layer Score [adjusted_final_layer_score]",
                "0.123456789012345678901234567890  ← primary score",
            ]
        ],
        subtitle=(
            "lower is better; adjusted_final_layer_score = final_layer_mse × max(0.1, C_m/B_m); "
            "failure → × 1.0 and this subtitle should not be truncated"
        ),
    )

    plain = render_document(
        title="WhestBench Report",
        blocks=[
            build_budget_breakdown_block(budget_section),
            build_score_block(score_section),
        ],
        output_format="plain",
        width=40,
    )

    for text in (
        "WhestBench Report",
        "Estimator Budget Breakdown",
        "Final Score",
        "lower is better",
    ):
        assert text in plain
    assert "metric | value" not in plain
    assert "namespace | total flops" not in plain
    assert max(len(line) for line in plain.splitlines()) < 200


def test_score_block_renders_primary_score_annotation():
    """The adjusted-MSE row's value cell carries the '← primary score' annotation."""
    from rich.console import Console

    from whestbench.presentation.adapters import _score_section
    from whestbench.presentation.blocks import build_score_block

    report = {
        "results": {
            "adjusted_final_layer_score": 0.245,
            "final_layer_mse": 0.220,
            "all_layers_mse": 0.178,
            "best_mlp_adjusted_final_layer_score": 0.0,
            "worst_mlp_adjusted_final_layer_score": 1.0,
            "mean_score_multiplier": 0.78,
            "mean_compute_utilization": 0.62,
            "n_failed_mlps": 0,
            "per_mlp": [{"adjusted_final_layer_score": 0.245}],
        }
    }
    section = _score_section(report)
    panel = build_score_block(section)
    console = Console(record=True, color_system=None, no_color=True, width=120)
    console.print(panel)
    rendered = console.export_text()
    assert "← primary score" in rendered


def test_score_block_renders_section_dividers():
    """Section dividers separate accuracy / range / efficiency groups."""
    from rich.console import Console

    from whestbench.presentation.adapters import _score_section
    from whestbench.presentation.blocks import build_score_block

    report = {
        "results": {
            "adjusted_final_layer_score": 0.245,
            "final_layer_mse": 0.220,
            "all_layers_mse": 0.178,
            "best_mlp_adjusted_final_layer_score": 0.0,
            "worst_mlp_adjusted_final_layer_score": 1.0,
            "mean_score_multiplier": 0.78,
            "mean_compute_utilization": 0.62,
            "n_failed_mlps": 0,
            "per_mlp": [{"adjusted_final_layer_score": 0.245}],
        }
    }
    section = _score_section(report)
    panel = build_score_block(section)
    console = Console(record=True, color_system=None, no_color=True, width=120)
    console.print(panel)
    rendered = console.export_text()
    divider_count = rendered.count("─" * 8)
    assert divider_count >= 2, f"Expected ≥2 dividers, got {divider_count}"


def test_budget_breakdown_renders_effective_compute_row():
    from rich.console import Console

    from whestbench.presentation.adapters import _breakdown_section
    from whestbench.presentation.blocks import build_budget_breakdown_block

    report = {
        "run_config": {"flop_budget": 10_000_000_000, "n_mlps": 1},
        "results": {
            "per_mlp": [
                {
                    "flops_used": 1_000_000_000,
                    "effective_compute": 6_000_000_000,
                    "combined_budget_exhausted": False,
                },
            ],
            "breakdowns": {
                "estimator": {
                    "flops_used": 1_000_000_000,
                    "flopscope_backend_time_s": 0.1,
                    "flopscope_overhead_time_s": 0.05,
                    "residual_wall_time_s": 0.5,
                    "by_namespace": {},
                },
            },
        },
    }
    section = _breakdown_section(
        report, breakdown_key="estimator", title="Estimator Budget Breakdown"
    )
    assert section is not None
    panel = build_budget_breakdown_block(section)
    console = Console(record=True, color_system=None, no_color=True, width=140)
    console.print(panel)
    rendered = console.export_text()
    assert "Effective Compute" in rendered
    assert "effective_compute" in rendered


def test_over_budget_table_renders_reason_column():
    from rich.console import Console

    from whestbench.presentation.adapters import _breakdown_section
    from whestbench.presentation.blocks import build_budget_breakdown_block

    report = {
        "run_config": {"flop_budget": 10_000_000_000, "n_mlps": 1},
        "results": {
            "per_mlp": [
                {
                    "mlp_index": 0,
                    "flops_used": 7_000_000_000,
                    "effective_compute": 12_000_000_000,
                    "combined_budget_exhausted": True,
                },
            ],
            "breakdowns": {
                "estimator": {
                    "flops_used": 7_000_000_000,
                    "flopscope_backend_time_s": 0.1,
                    "flopscope_overhead_time_s": 0.05,
                    "residual_wall_time_s": 0.5,
                    "by_namespace": {},
                },
            },
        },
    }
    section = _breakdown_section(
        report, breakdown_key="estimator", title="Estimator Budget Breakdown"
    )
    assert section is not None
    panel = build_budget_breakdown_block(section)
    console = Console(record=True, color_system=None, no_color=True, width=140)
    console.print(panel)
    rendered = console.export_text()
    assert "reason" in rendered
    assert "COMBINED" in rendered
