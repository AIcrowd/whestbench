"""Layer diagnostics tab view for the Textual dashboard."""

from __future__ import annotations

from ..state import DashboardState


def render_layers_view(state: DashboardState) -> str:
    """Render the layer diagnostics deep-dive tab."""

    points = ", ".join(f"{value:.6f}" for value in state.derived.layer_mse_mean_by_index)
    return (
        "Layer Analysis\n\n"
        "Layer Diagnostics\n"
        "- mse_mean_by_layer trend\n"
        f"- layer means: [{points}]\n"
    )
