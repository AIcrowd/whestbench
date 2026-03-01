from __future__ import annotations

import circuit_estimation.textual_dashboard.plots as plot_module
from circuit_estimation.textual_dashboard.plots import (
    build_budget_frontier_plot,
    build_budget_runtime_plot,
    build_layer_trend_plot,
    build_profile_memory_plot,
    build_profile_runtime_plot,
)


def test_budget_frontier_plot_returns_chart_and_legend() -> None:
    chart, legend = build_budget_frontier_plot(
        budgets=[10, 100],
        adjusted_mse=[0.22, 0.036],
        mse_mean=[0.2, 0.04],
        width=60,
        height=12,
    )

    assert chart.strip()
    assert "adjusted_mse" in legend


def test_plot_builder_fallback_when_plotext_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        plot_module,
        "_render_plotext_chart",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    chart, legend = build_budget_frontier_plot(
        budgets=[10, 100],
        adjusted_mse=[0.22, 0.036],
        mse_mean=[0.2, 0.04],
        width=60,
        height=12,
    )

    assert chart.strip()
    assert "plot unavailable" in legend.lower()


def test_all_plot_builders_return_non_empty_output() -> None:
    chart1, legend1 = build_budget_runtime_plot(
        budgets=[10, 100, 1000],
        time_ratio=[1.2, 0.9, 0.8],
        effective_time=[0.012, 0.018, 0.031],
        width=60,
        height=12,
    )
    chart2, legend2 = build_layer_trend_plot(
        mse_by_layer=[0.2, 0.18, 0.14, 0.11, 0.08],
        width=60,
        height=12,
    )
    chart3, legend3 = build_profile_runtime_plot(
        wall_s=[0.01, 0.013, 0.022],
        cpu_s=[0.009, 0.011, 0.02],
        width=60,
        height=12,
    )
    chart4, legend4 = build_profile_memory_plot(
        rss_mb=[120.0, 122.0, 127.5],
        peak_mb=[130.0, 132.5, 138.0],
        width=60,
        height=12,
    )

    for chart, legend in ((chart1, legend1), (chart2, legend2), (chart3, legend3), (chart4, legend4)):
        assert chart.strip()
        assert legend.strip()
