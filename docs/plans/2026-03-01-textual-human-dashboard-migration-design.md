# Textual Human Dashboard Migration Design

Date: 2026-03-01  
Status: Approved

## 1. Goal

Replace the current default human-mode Rich dashboard with a full-screen Textual app that feels premium, modern, and easier to navigate, while preserving machine-mode JSON output for automation.

Primary UX requirement:

1. The first screen must provide high-level summary plus interesting plots.
2. Everything visible in current human mode must also be visible in this first view.
3. Additional tabs provide deeper drill-down.

## 2. Product Decisions (Locked)

1. Human mode is a full-screen interactive Textual TUI by default.
2. `--agent-mode` is retained for JSON-only machine output.
3. Remove `--detail`, `--profile`, and `--show-diagnostic-plots` from CLI.
4. Human mode always collects full profiling data for the Performance tab.
5. If Textual is unavailable (non-TTY/capability/init failure), auto-fallback to static report rendering.
6. Visual direction is scientific-editorial (light, structured, high-contrast, publication feel).

## 3. Information Architecture

### 3.1 Tabs

1. `Summary` (default)
2. `Budgets`
3. `Layers`
4. `Performance`
5. `Data`

### 3.2 Summary Tab Contract

`Summary` must include:

1. A high-level executive strip (score, status chips, run identity).
2. A curated "interesting plots" area near the top.
3. All metrics/sections currently shown in legacy human mode.
4. Short insights/recommendations block.

This tab is scrollable and ordered for quick read first, detailed inspection second.

### 3.3 Deep-Dive Tabs

1. `Budgets`: expanded budget table, trends, sortable/focus interactions.
2. `Layers`: layer trajectories, quantile diagnostics, anomaly surfacing.
3. `Performance`: per-call runtime and memory diagnostics.
4. `Data`: structured/raw payload inspection.

## 4. Architecture

### 4.1 Core Modules

1. `DashboardApp` (Textual `App`): shell, theme, key bindings, tab routing.
2. `DashboardState`: normalized report + derived metrics + UI state.
3. View modules per tab: summary, budgets, layers, performance, data.
4. Shared widget primitives: metric cards, status chips, panels, plot blocks, insight lists.

### 4.2 Data Ownership

1. Scoring produces one report payload.
2. Derived metrics are computed once in a centralized state/adapter layer.
3. Views read state; views do not recompute shared aggregates.

### 4.3 CLI Routing

1. Parse args.
2. Run scorer.
3. If `--agent-mode`: emit JSON only.
4. Else: launch Textual app if supported, otherwise render static fallback.

## 5. UX and Interaction

### 5.1 Keyboard Model

1. Tabs via `1-5` and standard tab navigation.
2. Global shortcuts: `q` quit, `?` help, `r` refresh/reload.

### 5.2 Visual System (Scientific Editorial)

1. Light background, sharp hierarchy, restrained accent palette.
2. Semantic colors:
   1. Green for primary score health.
   2. Blue for accuracy signals.
   3. Amber for runtime/performance.
   4. Red for warnings/errors.
3. Clean spacing rhythm and clear focus states for keyboard use.
4. Subtle transitions only (tab/view change, staged load), no noisy animation.

## 6. Compatibility and Fallback

1. `--agent-mode` contract remains JSON-only.
2. Human-mode fallback prints a short stderr notice and renders static report to stdout.
3. Unsupported-terminal cases must not break automated workflows.

## 7. Rollout Plan

### Phase 1 (Core)

1. Textual default app with all five tabs.
2. Summary tab includes all legacy human-mode information.
3. CLI simplified to `--agent-mode` only.
4. Static fallback path implemented and tested.

### Phase 2 (Premium)

1. Command palette and richer keyboard workflows.
2. Drill-down overlays/modals.
3. Interaction and motion polish pass.

## 8. Verification Strategy

1. CLI tests for new argument surface and routing behavior.
2. State/derivation tests for deterministic computed metrics.
3. Textual widget/app tests for tab navigation and required anchors.
4. Regression checks for agent JSON path.
5. Fallback tests for non-TTY/unsupported environments.

## 9. Acceptance Criteria

1. Default human mode launches Textual UI.
2. Summary tab contains all information from current human mode.
3. Tabs provide meaningful deeper analysis.
4. CLI no longer exposes `--detail`, `--profile`, or `--show-diagnostic-plots`.
5. `--agent-mode` remains stable and machine-friendly.
6. Textual fallback is automatic and safe.
7. Test suite coverage is updated for new contracts.
