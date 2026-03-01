"""Shared plotext chart builders for Textual dashboard panes."""

from __future__ import annotations

import re
from statistics import fmean

try:
    import plotext as _plotext  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - optional dependency
    _plotext = None


def build_budget_frontier_plot(
    *,
    budgets: list[int],
    adjusted_mse: list[float],
    mse_mean: list[float],
    width: int,
    height: int,
) -> tuple[str, str]:
    chart = _safe_render_plotext_chart(
        x=[float(value) for value in budgets],
        series=[
            ("adjusted_mse", adjusted_mse, "cyan+"),
            ("mse_mean", mse_mean, "magenta+"),
        ],
        x_label="budget",
        y_label="score",
        width=width,
        height=height,
        x_scale="log",
    )
    return _chart_with_legend(
        chart,
        [
            ("adjusted_mse", adjusted_mse),
            ("mse_mean", mse_mean),
        ],
    )


def build_budget_runtime_plot(
    *,
    budgets: list[int],
    time_ratio: list[float],
    effective_time: list[float],
    width: int,
    height: int,
) -> tuple[str, str]:
    chart = _safe_render_plotext_chart(
        x=[float(value) for value in budgets],
        series=[
            ("call_time_ratio_mean", time_ratio, "yellow+"),
            ("call_effective_time_s_mean", _normalize(effective_time), "green+"),
        ],
        x_label="budget",
        y_label="runtime",
        width=width,
        height=height,
        x_scale="log",
    )
    return _chart_with_legend(
        chart,
        [
            ("call_time_ratio_mean", time_ratio),
            ("call_effective_time_s_mean", effective_time),
        ],
    )


def build_layer_trend_plot(
    *,
    mse_by_layer: list[float],
    width: int,
    height: int,
) -> tuple[str, str]:
    x = [float(index) for index in range(len(mse_by_layer))]
    chart = _safe_render_plotext_chart(
        x=x,
        series=[("mse_by_layer", mse_by_layer, "cyan+")],
        x_label="layer",
        y_label="mse",
        width=width,
        height=height,
    )
    return _chart_with_legend(chart, [("mse_by_layer", mse_by_layer)])


def build_profile_runtime_plot(
    *,
    wall_s: list[float],
    cpu_s: list[float],
    width: int,
    height: int,
) -> tuple[str, str]:
    x = [float(index) for index in range(max(len(wall_s), len(cpu_s)))]
    wall = _pad_to_len(wall_s, len(x))
    cpu = _pad_to_len(cpu_s, len(x))
    chart = _safe_render_plotext_chart(
        x=x,
        series=[
            ("wall_time_s", wall, "cyan+"),
            ("cpu_time_s", cpu, "magenta+"),
        ],
        x_label="call_index",
        y_label="seconds",
        width=width,
        height=height,
    )
    return _chart_with_legend(chart, [("wall_time_s", wall_s), ("cpu_time_s", cpu_s)])


def build_profile_memory_plot(
    *,
    rss_mb: list[float],
    peak_mb: list[float],
    width: int,
    height: int,
) -> tuple[str, str]:
    x = [float(index) for index in range(max(len(rss_mb), len(peak_mb)))]
    rss = _pad_to_len(rss_mb, len(x))
    peak = _pad_to_len(peak_mb, len(x))
    chart = _safe_render_plotext_chart(
        x=x,
        series=[
            ("rss_mb", rss, "cyan+"),
            ("peak_rss_mb", peak, "magenta+"),
        ],
        x_label="call_index",
        y_label="memory_mb",
        width=width,
        height=height,
    )
    return _chart_with_legend(chart, [("rss_mb", rss_mb), ("peak_rss_mb", peak_mb)])


def _render_plotext_chart(
    *,
    x: list[float],
    series: list[tuple[str, list[float], str]],
    x_label: str,
    y_label: str,
    width: int,
    height: int,
    x_scale: str | None = None,
) -> str | None:
    if _plotext is None or not x:
        return None

    valid_series = [
        (label, values, color)
        for label, values, color in series
        if len(values) == len(x) and len(values) > 0
    ]
    if not valid_series:
        return None

    try:
        _plotext.clear_data()
        _plotext.clear_figure()
        _plotext.theme("clear")
        _plotext.plotsize(max(36, width), max(8, height))
        _plotext.canvas_color("black")
        _plotext.axes_color("white")
        _plotext.ticks_color("white")

        if x_scale is not None:
            _plotext.xscale(x_scale)

        for _label, values, color in valid_series:
            if len(x) <= 12:
                scatter_fn = getattr(_plotext, "scatter", None)
                if callable(scatter_fn):
                    scatter_fn(x, values, color=color, marker="●")
                else:
                    _plotext.plot(x, values, color=color, marker="●")
            else:
                _plotext.plot(x, values, color=color, marker="hd")

        _plotext.xlabel(x_label)
        _plotext.ylabel(y_label)
        _plotext.grid(True, False)
        return _sanitize_plotext_ansi(str(_plotext.build()))
    except Exception:  # pragma: no cover - terminal backends vary by environment
        return None
    finally:
        if _plotext is not None:
            try:
                _plotext.clear_data()
                _plotext.clear_figure()
            except Exception:  # pragma: no cover - best effort
                pass


def _safe_render_plotext_chart(
    *,
    x: list[float],
    series: list[tuple[str, list[float], str]],
    x_label: str,
    y_label: str,
    width: int,
    height: int,
    x_scale: str | None = None,
) -> str | None:
    """Guard plot rendering to ensure callers always degrade gracefully."""

    try:
        return _render_plotext_chart(
            x=x,
            series=series,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            x_scale=x_scale,
        )
    except Exception:  # pragma: no cover - defensive wrapper for monkeypatch/fault injection
        return None


def _chart_with_legend(
    chart: str | None,
    legend_series: list[tuple[str, list[float]]],
) -> tuple[str, str]:
    if chart is None:
        fallback_rows = []
        for label, values in legend_series:
            fallback_rows.append(
                f"{label}: p05={_percentile(values, 0.05):.6f} min={min(values) if values else 0.0:.6f} "
                f"mean={fmean(values) if values else 0.0:.6f} max={max(values) if values else 0.0:.6f} "
                f"p95={_percentile(values, 0.95):.6f}"
            )
        chart = "\n".join(fallback_rows) if fallback_rows else "no data"
        return chart, "plot unavailable; using numeric fallback"

    legend_parts = []
    for label, values in legend_series:
        start = min(values) if values else 0.0
        end = max(values) if values else 0.0
        legend_parts.append(f"{label}: {start:.6f} -> {end:.6f}")
    return chart, " | ".join(legend_parts)


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _pad_to_len(values: list[float], size: int) -> list[float]:
    if not values:
        return [0.0 for _ in range(size)]
    if len(values) >= size:
        return values[:size]
    return values + [values[-1] for _ in range(size - len(values))]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _sanitize_plotext_ansi(chart: str) -> str:
    without_bg = re.sub(r"\x1b\[48;[0-9;]*m", "", chart)
    return re.sub(r"\x1b\[49m", "", without_bg)
