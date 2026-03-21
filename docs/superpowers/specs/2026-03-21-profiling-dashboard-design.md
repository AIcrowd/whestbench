# Profiling Dashboard Design Spec

## Overview

A Python script that generates a self-contained, interactive HTML dashboard from profiling results. The dashboard uses React 18 (via CDN) with inline components to visualize backend performance across hardware configurations, with drill-down modals for detailed timing and operation breakdowns.

## Goals

- Intuitively explore and compare profiling results across backends and hardware configs
- Work with both multi-config cloud runs (`combined.json`) and single-config local runs (`output.json`)
- Produce a single HTML file — shareable, committable, works offline. The generator fetches CDN libraries at generation time and inlines them, so the output has zero runtime dependencies.

## Input Formats

### Multi-config (cloud runs)
File: `combined.json` from `collect_results.py`

```json
{
  "run_id": "2026-03-20-213311-6150c93-dirty",
  "git_commit": "6150c93",
  "git_dirty": true,
  "collected_at": "2026-03-20T21:51:41+00:00",
  "configs": {
    "compute-2xlarge": {
      "hardware": { "cpu_count_logical": 16, "ram_total_bytes": 66594476032, ... },
      "backend_versions": { "numpy": "2.2.6", "jax": "0.6.2", ... },
      "skipped_backends": {},
      "correctness": [ { "backend": "numpy", "passed": true, "error": "" }, ... ],
      "timing": [
        {
          "backend": "jax",
          "operation": "run_mlp",
          "width": 256, "depth": 32, "n_samples": 100000,
          "times": [0.2032, 0.2051, 0.2089],
          "median_time": 0.2051,
          "speedup_vs_numpy": 2.35
        },
        ...
      ]
    },
    ...
  }
}
```

`run_mlp_profiled` entries include a `breakdown` field:
```json
{
  "breakdown": {
    "matmul_per_layer": [0.00175, 0.00167, ...],
    "relu_per_layer": [0.00446, 0.00158, ...],
    "total_matmul": 0.00672,
    "total_relu": 0.00912,
    "overhead": 0.000019,
    "total": 0.01585,
    "matmul_pct": 42.4,
    "relu_pct": 57.5,
    "overhead_pct": 0.1
  }
}
```

### Single-config (local runs)
File: `output.json` from `nestim profile-simulation --output output.json`

Same structure as one entry in `configs`, with top-level `hardware`, `backend_versions`, `correctness`, and `timing` fields. The generator wraps this into the multi-config format with config name derived from hostname or "local".

## CLI Interface

```bash
# From cloud run results
python -m profiling.generate_dashboard --run-id 2026-03-20-213311-6150c93-dirty

# From local profiling output
python -m profiling.generate_dashboard --input output.json

# Custom output path
python -m profiling.generate_dashboard --input output.json --output my-report.html
```

**Arguments:**
- `--run-id` — load from `profiling/results/{run-id}/combined.json`
- `--input` — path to a `combined.json` or single-config `output.json`
- `--output` — output HTML path (default: `profiling/results/{run-id}/dashboard.html` or `./dashboard.html` for `--input`)

Exactly one of `--run-id` or `--input` is required.

## Dashboard Layout

### Header Bar
- White background with bottom border (`#D9DCDC`), matching network-explorer header style
- Left: "nestim Profiling Dashboard" in Montserrat 700, gray-900
- Right: Run metadata — run ID, git commit (with coral "dirty" badge if applicable), timestamp in gray-600
- Run selector is out of scope for v1 (each HTML file contains one run's data)

### Section 1: Speedup Heatmap
The primary visualization. A table with:
- **Rows**: Hardware configs, labeled as `config-name (Xc / YG)` using cpu_count_logical and ram_total_bytes
- **Columns**: Backends (numpy, pytorch, jax, cython, numba, scipy)
- **Cell values**: Speedup vs numpy (e.g., `2.35`)
- **Cell colors**: Green gradient for >1.0 (faster), red gradient for <1.0 (slower), grey for 1.0
- **Best value per row**: Bold text

**Filters** (dropdowns above the heatmap):
- Operation: `run_mlp` (default), `output_stats`, `run_mlp_profiled`
- Width: all available values, default to largest
- Depth: all available values, default to largest
- N Samples: all available values, default to largest

**Correctness indicator**: If a backend failed correctness checks for a config, show a warning icon (⚠) on the heatmap cell. Hovering shows the error message.

**Interaction**: Click a cell to open the detail modal.

**Single-config mode**: Only one row. Still useful for comparing backends.

### Section 2: Cell Detail Modal
Opens when clicking a heatmap cell. White modal with gray-200 border, 8px radius, overlay backdrop:

**Header**: Backend name (Montserrat 700) + speedup badge (success green "Nx faster" / coral "Nx slower") + close button

**Filter row**: Gray-50 background row with dropdowns to change config, width, depth, n_samples without closing. Dropdowns use gray-100 background, gray-200 border, coral focus ring.

**Left column — Timing**:
- Large median time display (e.g., "0.2051s")
- "median of N runs" subtitle
- Horizontal bar chart showing each individual run time, median run highlighted in green
- Standard deviation and coefficient of variation

**Right column — Operation Breakdown** (from `run_mlp_profiled` data):
- Stacked horizontal bar: MatMul (dark slate `#334155`) | ReLU (coral `#F0524D`) | Overhead (gray-200)
- Legend with absolute time and percentage for each
- Per-layer sparkline bar chart showing MatMul time per layer (layer 1 to layer N)

If no `run_mlp_profiled` data exists for the selected params, show "No profiled data for this configuration" message.

**Footer**: Visual bar comparison vs numpy — two horizontal bars showing relative time.

### Section 3: CPU Scaling Chart (multi-config only)
Line chart using Recharts:
- **X-axis**: Config names as discrete labeled points, sorted by vCPU count
- **Y-axis**: Median execution time (seconds)
- **Lines**: One per backend, color-coded
- **Tooltip**: Shows exact time, speedup, config name, and hardware specs on hover
- **Filter**: Config family toggle (compute / general / all) since compute-optimized and general-purpose configs at the same vCPU count may differ
- Dropdown to select operation, width, depth, n_samples

Hidden in single-config mode.

### Section 4: Raw Data Table
Sortable table of all timing entries:
- Columns: Config, Backend, Operation, Width, Depth, N Samples, Median Time, Speedup vs NumPy
- Click column headers to sort
- Filter row at top with dropdowns for each categorical column
- Expandable rows showing hardware info and backend versions

## Technical Architecture

### File: `profiling/generate_dashboard.py`

**Responsibilities:**
1. Parse CLI args (`--run-id` or `--input`, `--output`)
2. Load and normalize input data (wrap single-config into multi-config format)
3. Generate HTML string with embedded React app
4. Write to output file

**HTML generation approach:**

The generator avoids JSX and Babel entirely by using `React.createElement` calls directly. This eliminates a ~3MB Babel dependency and removes browser-side transpilation latency. The Python script:
1. Fetches React, ReactDOM, and Recharts UMD bundles from CDN at generation time
2. Caches them locally in `profiling/.lib-cache/` for subsequent runs
3. Inlines the library source + component code + CSS + data into a single HTML file

```html
<!DOCTYPE html>
<html>
<head>
  <title>nestim Profiling Dashboard</title>
  <script>/* inlined react.production.min.js */</script>
  <script>/* inlined react-dom.production.min.js */</script>
  <script>/* inlined Recharts.min.js */</script>
  <style>/* all CSS inlined here */</style>
</head>
<body>
  <div id="root"></div>
  <script>window.__PROFILING_DATA__ = {/* JSON data */};</script>
  <script>/* React components using React.createElement */</script>
</body>
</html>
```

### React Components

1. **`App`** — top-level state management, data loading from `window.__PROFILING_DATA__`
2. **`Header`** — run metadata display
3. **`SpeedupHeatmap`** — filterable heatmap table, click handler to open modal
4. **`CellDetailModal`** — full detail view with dropdowns, timing bars, breakdown charts
5. **`CPUScalingChart`** — Recharts LineChart (hidden in single-config mode)
6. **`DataTable`** — sortable/filterable raw data table

### Design System (matching network-explorer)

Follow the same design language as `tools/network-explorer/` for visual consistency.

**Typography** (Google Fonts, inlined):
- Body/UI: `Inter`, 13px base, weights 400/500/600
- Headers: `Montserrat`, weight 600/700, letter-spacing -0.02em
- Monospace (values, code): `IBM Plex Mono`, weights 400/500

**Color Palette:**
- **Coral (primary accent)**: `#F0524D` (hover: `#EE403A`, pressed: `#D23934`, light: `#FEF2F1`)
- **Page background**: `#FFFFFF`
- **Card/Panel background**: `#FFFFFF` with `1px solid #D9DCDC` border
- **Subtle background**: `#F8F9F9` (gray-50)
- **Light background**: `#F1F3F5` (gray-100)
- **Borders**: `#D9DCDC` (gray-200)
- **Labels/disabled**: `#AAACAD` (gray-400)
- **Secondary text**: `#5D5F60` (gray-600)
- **Primary text**: `#292C2D` (gray-900)
- **Success**: `#23B761`
- **Warning**: `#FA9E33`

**Heatmap colors:**
- **Faster (>1.0)**: linear interpolation from `#c8e6c9` (1.1x) to `#23B761` (2.5x+, clamped)
- **Slower (<1.0)**: linear interpolation from `#FEF2F1` (0.9x) to `#F0524D` (0.3x, clamped)
- **Neutral (0.9–1.1)**: `#F1F3F5` (gray-100), linear blend toward green/red at edges

**Operation breakdown:**
- **MatMul**: `#334155` (dark slate, matching network-explorer chart palette)
- **ReLU**: `#F0524D` (coral)
- **Overhead**: `#D9DCDC` (gray-200)

**Backend colors (consistent throughout):**
- numpy: `#94A3B8` (slate, matching chart-2)
- pytorch: `#e65100`
- jax: `#334155` (dark slate)
- cython: `#23B761` (success green)
- numba: `#6a1b9a`
- scipy: `#00838f`

**Shared patterns:**
- Border radius: 8px (standard), 20px (pills/badges)
- Shadows: `0 2px 6px rgba(0,0,0,0.08)` (cards), `0 8px 24px rgba(0,0,0,0.12)` (modals)
- Section headers: 11px uppercase, letter-spacing 0.08em, gray-400
- Table headers: 10px uppercase, gray-400, weight 600
- Table rows: alternate gray-50 background on even rows
- Hover: `translateY(-1px)` lift with enhanced shadow
- Focus: `0 0 0 3px #FEF2F1` ring with coral border
- Animations: 0.2s transitions, entrance animations with fade+slide (0.25s ease-out)

**Modal styling:**
- White background with gray-200 border, 8px radius
- Overlay: `rgba(0, 0, 0, 0.3)` backdrop
- Header: Montserrat 700, gray-900 text, coral accent badge
- Filter dropdowns: gray-100 background, gray-200 border, coral focus ring

## Data Flow

```
combined.json or output.json
  → generate_dashboard.py (Python)
    → normalize to multi-config format
    → JSON.stringify and embed as window.__PROFILING_DATA__
    → inline React components + CSS
    → write dashboard.html

dashboard.html (opened in browser)
  → React reads window.__PROFILING_DATA__
  → renders Header, SpeedupHeatmap, CPUScalingChart, DataTable
  → user clicks heatmap cell → CellDetailModal opens
  → user changes modal dropdowns → modal content updates
```

## Edge Cases

- **Single-config input**: Hide CPU scaling chart, heatmap has one row, modal config dropdown disabled
- **Missing `run_mlp_profiled` data**: Show "No profiled breakdown available" in modal right panel
- **Missing backends**: Some configs may not have all 6 backends. Show "—" in heatmap, skip in charts.
- **No data for selected filter combo**: Show "No data for this combination" message
- **Large datasets**: Standard preset produces ~540 timing entries per config × 9 configs = ~4,860 entries. Embedded JSON will be ~500KB — acceptable for a single HTML file.
- **Skipped backends**: If `skipped_backends` has entries, show "skipped" in heatmap cell with reason on hover (distinct from missing data "—").
- **Failed correctness with valid timing**: Show timing data but with a ⚠ warning badge. Tooltip explains the correctness failure.
- **Empty config (no timing data)**: Show the row with all cells as "—" and a note "no data" rather than hiding it.
