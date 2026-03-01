# Human Dashboard Redesign (Tri-Objective, Multi-Column)

Date: 2026-03-01  
Status: Approved for planning

## 1. Goal

Redesign the default human CLI dashboard so it is simultaneously strong at:

1. Submission readiness: quick decision support on whether a run looks competitive.
2. Learning/interpretability: clear signal on how budgets and layers affect outcomes.
3. Performance debugging: practical runtime and memory diagnostics when profiling is enabled.

The interface should look professional, use a true multi-column/multi-pane structure, and remain robust across different terminal widths.

## 2. Core Decisions

1. Keep one balanced, high-quality default human dashboard (no template split by persona).
2. Use a multi-column layout with nested Rich panes/tables.
3. Preserve agent-mode output contract as pretty JSON only.
4. Show hardware/runtime metadata only from fields already captured by scorer.
5. Keep plots informative by using different rendering policies for sparse vs dense series.

## 3. Approved Information Architecture

### 3.1 Wide layout (primary target)

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          CIRCUIT ESTIMATION DASHBOARD                                               │
├───────────────────────────────────────────────┬───────────────────────────────────────────────┬────────────────────┤
│ Run Context                                   │ Readiness Scorecard                           │ Hardware & Runtime │
│ - Started/Finished/Duration                   │ - Final Score (Adjusted MSE Mean)             │ - Hostname         │
│ - Circuits/Samples/Width/Depth/Layers         │ - Best/Worst budget score                      │ - OS / Release     │
│ - Budgets/Time tolerance                      │ - Score spread + quick verdict                 │ - Platform/Machine │
│                                               │                                                 │ - Python version   │
├───────────────────────────────────────────────┼───────────────────────────────────────────────┼────────────────────┤
│ Budget Table                                  │ Budget Frontier Plot                           │ Layer Diagnostics  │
│ - score, mse, time ratio, eff time            │ - score vs avg_mse vs budget                   │ - p05/min/p95/max  │
│ - best row highlighted                         │ - runtime-vs-budget companion plot             │ - layer trend plot │
├───────────────────────────────────────────────┼───────────────────────────────────────────────┼────────────────────┤
│ Profile Summary (if --profile)                │ Profile Runtime Plot                           │ Profile Memory Plot│
│ - calls, wall/cpu/rss/peak                    │ - call_index vs wall/cpu                       │ - call_index vs rss│
│ - p05/p95/min/max                             │                                               │ - peak_rss         │
└───────────────────────────────────────────────┴───────────────────────────────────────────────┴────────────────────┘
```

### 3.2 Adaptive layout rules

1. Wide (`>=120` cols): 3-column layout shown above.
2. Medium (`90-119` cols): collapse to 2 columns, keep same narrative order.
3. Narrow (`<90` cols): stack panes in one column, same order as wide layout.

No pane should disappear due to width; only layout changes.

## 4. Pane-Level Metric Contract

### 4.1 Run Context (operational facts)

- Run Started/Finished/Duration
- Number of Circuits
- Samples per Circuit
- Circuit Width / Wire Count
- Maximum Depth
- Layer Count
- Budgets
- Time Tolerance

### 4.2 Readiness Scorecard (decision first)

- Final Score (Adjusted MSE Mean) `[final_score]` (primary line)
- MSE Mean `[mse_mean]`
- Best Budget Score `[best_budget_score]`
- Worst Budget Score `[worst_budget_score]`
- Score Spread `[score_spread] = worst - best`
- subtle footnote: lower score is better

### 4.3 Hardware & Runtime (existing captured fields only)

From `run_meta.host` only:

- `host.hostname`
- `host.os`
- `host.os_release`
- `host.platform`
- `host.machine`
- `host.python_version`

No new machine probing in this redesign.

### 4.4 Budget Intelligence

- Budget table:
  - budget
  - score
  - MSE mean
  - time-ratio mean
  - effective-time mean
- Plot pane:
  - budget frontier (accuracy metrics only)
  - budget runtime (runtime metrics only; normalized if mixed scale)

### 4.5 Layer Intelligence

- Layer stats table (p05/min/p95/max/mean) for:
  - `mse_by_layer`
  - `adjusted_mse_by_layer`
  - `time_ratio_by_layer`
- Layer plots:
  - accuracy trends
  - runtime trend (separate axis/plot)

### 4.6 Profile Lane (conditional)

Visible only when `profile_calls` exists.

- Profile summary table
- Profile distribution table (p05/p95/min/max)
- Runtime plot (`wall_time_s`, `cpu_time_s`)
- Memory plot (`rss_bytes`, `peak_rss_bytes`, de-duplicate if equal)

If profiling is disabled, the layout closes the profile row cleanly.

## 5. Visual System Rules

1. Human label + `[code_key]` pattern for skimmable context.
2. External legends only; no in-plot legend overlap.
3. No sparkline-only diagnostics where full plot/table exists.
4. Semantic colors are consistent across panes:
   - green: primary/better score
   - cyan: error metrics
   - yellow/magenta: runtime/perf series
   - red: warning/error conditions
5. Numeric format policy:
   - score-like: fixed high precision
   - seconds: moderate precision
   - bytes: readable precision

## 6. Plot Rendering Policy

1. Sparse X-series (few points, e.g., budgets): point-first rendering.
2. Dense X-series (many points, e.g., layer/call-index): line rendering.
3. Keep different scales separated (or explicitly normalized and labeled).
4. Always provide non-plot fallback table if plotting backend fails.

## 7. Implementation Structure

1. `render_human_report(report)` remains the orchestration entrypoint.
2. Build panes as pure functions returning `Panel`.
3. Build a width-aware layout spec (`wide`/`medium`/`narrow`).
4. Compose rows with `Columns`, and internal pane content with `Group`.
5. Keep plotting isolated in a single helper to centralize style policy.

## 8. Verification Strategy

1. Pane presence tests for section titles and key anchors.
2. Hardware pane tests must assert only currently captured host fields.
3. Plot-policy tests:
   - sparse path behavior,
   - dense path behavior,
   - fallback behavior.
4. CLI smoke coverage:
   - default human mode
   - `--profile`
   - `--agent-mode`
   - `--detail full`
5. Manual visual pass at three widths (`>=120`, `90-119`, `<90`).

## 9. Acceptance Criteria

1. One run output communicates readiness, interpretability, and performance clearly.
2. Dashboard is genuinely multi-column/multi-pane on wide terminals.
3. Hardware/runtime pane uses existing captured metadata only.
4. Sparse budget visuals are readable and non-chunky.
5. Agent JSON contract remains unchanged.
6. Lint/type/test suite remains green.
