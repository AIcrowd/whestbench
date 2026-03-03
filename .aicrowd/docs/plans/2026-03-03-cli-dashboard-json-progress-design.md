# CLI JSON Contract + Append-Only Dashboard Design

**Date:** 2026-03-03

## Goal

Fix CLI invocation and output contracts so `cestim --json` is always machine-parseable JSON, while human mode gets an append-only, professional dashboard with early context visibility and live progress feedback.

## Confirmed Requirements

- `--json` must never show the dashboard and must always print JSON to `stdout`.
- Human mode prints hints and all dashboard content to `stdout`.
- Completion flow is append-only (no redraw/reflow after progress completion).
- `Run Context` and `Hardware & Runtime` panes must always be adjacent (side-by-side).
- Estimator identity line must include both estimator class and estimator file path.
- Default human mode uses rich rendering and rich progress (`tqdm.rich`).
- `--disable-rich` enables plain dashboard rendering plus classic non-rich `tqdm` progress.
- `Scores` replaces `Readiness Scorecard`.

## Root Cause Summary

Installed `cestim` entrypoint calls `main()` with `argv=None`. Current `main()` uses `list(argv or [])`, which discards actual CLI flags (`sys.argv[1:]`). This causes `cestim --json` to execute default human dashboard mode and long scoring runs.

## Design Overview

### 1) Output/Mode Contract

- `main(argv=None)` resolves args from `sys.argv[1:]`.
- JSON mode (`--json`):
  - no dashboard rendering
  - no progress bars
  - no hint lines
  - final JSON only on `stdout`
- Human mode:
  - startup block + hints
  - pre-run side-by-side context panes
  - estimator identity line
  - tqdm progress
  - append final results sections

### 2) Append-Only Rendering Phases

#### Phase A (pre-run)

- Print report title line (centered)
- Print hints
- Print `Run Context` and `Hardware & Runtime` side-by-side
- Print estimator identity line:
  - legacy mode: built-in class + module path if available
  - participant `run`: estimator class + resolved estimator file path

#### Phase B (during run)

- Progress bar over total work units (`len(budgets) * n_circuits`) using:
  - rich tqdm in default mode
  - classic tqdm with `--disable-rich`

#### Phase C (post-run append)

Append sections in order:
1. `Scores`
2. `Budget` (table)
3. `Layer Diagnostics` (table)
4. `Profile` (if profile data present)
5. Optional plot panes if `--show-diagnostic-plots`
   - budget plots
   - layer trend plot
   - profile runtime+memory plots

## Full-Pane Wireframe (all extras on)

Assume: `cestim --profile --show-diagnostic-plots`

```text
[Phase A]
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Circuit Estimation Report                         │
└──────────────────────────────────────────────────────────────────────────────┘
Use --json for machine-readable output.
Use --show-diagnostic-plots to include diagnostic plot panes.
Use --disable-rich for plain dashboard + classic tqdm.

┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│ Run Context                          │  │ Hardware & Runtime                │
│ Started: ...                         │  │ Host: ...                         │
│ Circuits: ...  Samples/Circuit: ...  │  │ OS/Release: ...                  │
│ Width: ...  Max Depth: ...           │  │ Platform/Arch: ...               │
│ Budgets: [...]  Tolerance: ...       │  │ Python: ...                      │
└──────────────────────────────────────┘  └───────────────────────────────────┘
Estimator: class=<...> path=<...>

[Phase B]
Scoring progress:  45%|███████████▍| 18/40 [00:02<00:03, 6.8it/s]

[Phase C append]
┌────────────────────────────────────── Scores ────────────────────────────────┐
│ Final Score | MSE Mean | Best Budget Score | Worst Budget Score             │
└───────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────── Budget ────────────────────────────────┐
│ [budget table]                                                                 │
│ ┌───────────────────────────────┐  ┌──────────────────────────────────────┐ │
│ │ Budget Frontier Plot          │  │ Budget Runtime Plot                  │ │
│ └───────────────────────────────┘  └──────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────── Layer Diagnostics ───────────────────────────┐
│ [layer table]                                                                  │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ Layer Trend Plot                                                         │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────── Profile ────────────────────────────────┐
│ ┌───────────────────────────────┐  ┌──────────────────────────────────────┐ │
│ │ Summary                       │  │ Distribution                         │ │
│ └───────────────────────────────┘  └──────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
┌───────────────────────────────┐  ┌──────────────────────────────────────────┐
│ Profile Runtime Plot          │  │ Profile Memory Plot                      │
└───────────────────────────────┘  └──────────────────────────────────────────┘
```

## No-Profile / No-Plot Wireframe

Assume: `cestim`

```text
[Phase A] title + hints + Run Context || Hardware & Runtime + estimator identity
[Phase B] progress bar
[Phase C append] Scores -> Budget(table only) -> Layer Diagnostics(table only)
```

## Technical Changes

- `src/circuit_estimation/cli.py`
  - fix argv default behavior
  - add `--disable-rich` for legacy and participant `run`
  - split pre-run and post-run rendering for human mode
  - print estimator identity (class + path)
  - wire tqdm progress callback behavior
- `src/circuit_estimation/scoring.py`
  - add optional progress callback invoked per completed circuit per budget
- `src/circuit_estimation/reporting.py`
  - remove in-dashboard hint lines
  - remove `Rich Dashboard` subtitle
  - rename score panel title to `Scores`
  - add helpers for pre-run panes and post-run section-only rendering (append flow)
- tests
  - add regression for `main(argv=None)` using `sys.argv`
  - assert JSON mode emits JSON-only on `stdout`
  - assert renamed panel and removed subtitle/hints from final dashboard output
  - assert estimator identity includes class + path for participant run
  - assert progress plumbing toggles rich/classic modes via `--disable-rich`

## Error Handling

- Maintain existing rich-fallback behavior in human mode.
- In `--disable-rich`, fallback is inherently plain output.
- JSON mode error payload contract remains unchanged and parseable.

## Validation Plan

- Targeted tests first:
  - `tests/test_cli.py`
  - `tests/test_cli_fallback.py`
  - `tests/test_reporting.py`
- then full: `pytest -q`
- then manual smoke:
  - `uv tool install -e .`
  - `cestim --json`
  - `cestim --profile --show-diagnostic-plots`
  - `cestim --disable-rich`
