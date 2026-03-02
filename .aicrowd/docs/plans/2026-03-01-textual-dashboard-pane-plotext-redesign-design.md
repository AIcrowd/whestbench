# Textual Dashboard Pane + Plotext Redesign Design

Date: 2026-03-01  
Status: Approved

## 1. Goal

Redesign the default Textual human dashboard into a genuinely usable, skimmable, pane-oriented interface with a dark/black visual baseline, while preserving machine JSON output in `--agent-mode`.

Primary product requirements:

1. Summary tab must include **all information from the legacy Rich human interface**.
2. Plot rendering in the Textual UI should use `plotext`.
3. Deep tabs (`Budgets`, `Layers`, `Performance`, `Data`) should provide meaningful drill-downs.
4. Exit must be frictionless on every run (`Esc`, `Ctrl+C`, or `q`).

## 2. Locked Decisions

1. Keep tabs: `Summary`, `Budgets`, `Layers`, `Performance`, `Data`.
2. Use `plotext` as the chart backend in Textual panes.
3. Keep all major plots visible by default (no collapsed-by-default sections).
4. Keep raw payload inspection in `Data` tab only (no full payload dump in Summary).
5. Responsive breakpoints:
   - `wide >= 160`: 3 columns
   - `medium 110-159`: 2 columns
   - `narrow < 110`: single-column stack

## 3. Summary Tab Contract (Legacy-Complete)

Summary uses a thin status strip + strict legacy matrix.

### 3.1 Thin status strip

- final score
- best budget score
- worst budget score
- score spread
- run mode badge + run timestamp

### 3.2 Row 1 (always present)

1. **Run Context**
   - started/finished/duration
   - circuits/samples/width/max depth/layer count
   - budgets/time tolerance
2. **Readiness Scorecard**
   - final score
   - mse mean
   - best/worst budget score
   - score spread
   - lower-is-better hint
3. **Hardware & Runtime**
   - hostname
   - os/release/platform
   - machine
   - python version

### 3.3 Row 2 (always present)

1. **Budget Table**
   - budget
   - adjusted_mse
   - mse_mean
   - call_time_ratio_mean
   - call_effective_time_s_mean
2. **Budget Frontier Plot** (`plotext`)
   - x: budget
   - y: adjusted_mse + mse_mean
3. **Budget Runtime Plot** (`plotext`)
   - x: budget
   - y: call_time_ratio_mean + call_effective_time_s_mean
   - normalized secondary series when scales diverge

### 3.4 Row 3

1. **Layer Diagnostics Table**
   - p05/min/p95/max/mean (MSE-by-layer aggregate)
2. **Layer Trend Plot** (`plotext`)
   - x: layer index
   - y: mean MSE by layer
3. **Profile Composite** (conditional)
   - profile summary table
   - runtime plot (`wall_time_s`, `cpu_time_s`)
   - memory plot (`rss_bytes`, `peak_rss_bytes` in MB)
   - if profile absent: explicit unavailable panel (not blank)

## 4. Deep Tab Contracts

### 4.1 Budgets

- expanded sortable budget table
- frontier and runtime plots
- concise budget tradeoff insight pane

### 4.2 Layers

- depth stats table + worst-layer highlights
- MSE trend plot + optional smoothed overlay
- depth-focused diagnostic insight pane

### 4.3 Performance

- call summary table (mean/p95/min/max where relevant)
- runtime trend plot
- memory trend plot
- outlier call list
- explicit empty state when profile data is missing

### 4.4 Data

- structured payload inspector only:
  - run_meta
  - run_config
  - results.by_budget_raw
  - profile_calls

## 5. Plot Architecture

1. Centralize plotting logic in one shared module (`plotext` wrappers).
2. Reuse proven chart/ansi helpers from prior Rich implementations where practical.
3. Each plot pane returns:
   - chart text
   - legend/range note
4. If plotting fails:
   - degrade to table summary + explicit “plot unavailable” note
   - never crash the UI.

## 6. Component Architecture

1. `DashboardApp`: tab orchestration, keybindings, responsive mode handling.
2. `DashboardState`: single source of normalized/derived metrics.
3. `views/*`: pane builders consuming state only.
4. `widgets.py`: reusable pane/card/layout primitives.
5. `plots.py` (new): shared `plotext` chart construction and fallback logic.

## 7. Interaction Model

1. Tab switching: `1-5` + native tab navigation.
2. Help: `?`.
3. Refresh/recompute UI state: `r`.
4. Quit: `Esc`, `Ctrl+C`, `q`.

## 8. Visual System (Dark Baseline)

1. Black/deep-navy background with high-contrast foreground text.
2. Semantic accents:
   - green for score health
   - cyan/blue for accuracy plots
   - amber/yellow for runtime
   - magenta for memory
   - red for warnings
3. Consistent pane borders, spacing rhythm, and scannable title hierarchy.

## 9. Verification Strategy

1. Summary legacy-coverage tests for all required panes and anchors.
2. Plot presence tests for all expected plot panes.
3. Plot-fallback tests (forced builder error path).
4. Responsive layout tests across wide/medium/narrow.
5. Interaction tests (`Esc`, `Ctrl+C`, `q`, `1-5`, `r`, `?`).
6. CLI compatibility checks:
   - default human mode
   - `--agent-mode`
   - static fallback path

## 10. Acceptance Criteria

1. Summary tab includes all legacy Rich human-mode information.
2. Dashboard is truly pane-based and skimmable (not text dump blocks).
3. `plotext` drives substantive charts in Summary + deep tabs.
4. Dark theme is readable and visually coherent.
5. Exiting is instant via `Esc`, `Ctrl+C`, or `q`.
6. JSON machine contract in `--agent-mode` remains unchanged.
