# Profiler CLI Output Redesign

**Date:** 2026-03-20
**Status:** Approved

## Problem

The profiler CLI output dumps 5 separate Rich tables sequentially (Hardware, Skipped Backends, Correctness Check, Timing Results, Primitive Breakdown). Even with the `super-quick` preset, the output is hard to navigate and doesn't surface the most important information — which backend is fastest and where time is being spent — without scrolling through context that's rarely the primary concern.

## Design: "Leaderboard + Compact Table" with Verbose Drill-down

### Default Output (no flags)

Three zones, top to bottom:

#### Zone 1 — Context (1-3 lines, plain text)

```
macOS arm64 · 16 cores · 64 GB · Python 3.10 · NumPy 2.2
Skipped: pytorch, numba, jax, cython (--backends-help for install info)
Correctness: numpy ✓  scipy ✓
```

Rules:
- Hardware condensed to a single line: `{OS} {arch} · {physical_cores} cores · {ram} GB · Python {ver} · NumPy {ver}`
- If `cpu_count_physical` is `None` (psutil unavailable), fall back to `cpu_count_logical`; if both are `None`, omit the cores field entirely.
- Skipped line: comma-separated names only. Omitted entirely if no backends are skipped.
- Correctness line: backend names with ✓/✗ marks. If any backend fails, show `⚠ FAIL: {name} (use --verbose for details)` instead of ✗ for that backend.

#### Zone 2 — Leaderboard (ranked list per dimension combo)

```
── Leaderboard (w=64 d=4 n=1,000) ──────────────────────────
  #1  numpy   0.0001s
  #2  scipy   0.0002s  (0.83×)
```

Rules:
- Ranked by `run_mlp` median time, fastest first.
- Baseline selection: if numpy is in the backend list, it is the baseline. Otherwise, the fastest backend becomes the baseline (shown without a multiplier).
- All non-baseline backends show `(Nx)` relative to baseline — green if >1.0x (faster), red if <1.0x (slower).
- Failed backends excluded from ranking.
- Single-backend runs: leaderboard zone is omitted (detail table is sufficient).
- Zero passed backends: show `No backends passed correctness checks. Use --verbose for error details.` instead of zones 2 and 3.
- Multiple dimension combos (standard/exhaustive presets) produce multiple leaderboard groups, each with a dimension label header. The grouping is per unique `(width, depth, n_samples)` triple.

#### Zone 3 — Compact Detail Table (one Rich table)

```
Backend  Dims         run_mlp   output_stats   Matmul  ReLU   Ovhd
numpy    64×4×10k     0.0001s   0.0009s        42.5%   57.0%  0.5%  ✓
scipy    64×4×10k     0.0002s   0.0009s        48.3%   51.1%  0.6%  ✓
```

Rules:
- One row per backend per dimension combo.
- Columns: Backend, Dims (compact `w×d×n` format with k/M suffixes), run_mlp, output_stats, Matmul%, ReLU%, Overhead%, Status (✓/✗).
- Rows sorted by run_mlp time (fastest first), matching leaderboard order.
- Multiple dimension combos separated by a dashed line between groups.
- Matmul/ReLU/Overhead percentages are derived from the `run_mlp_profiled` operation's `PrimitiveBreakdown` data (`overhead_pct = overhead / total * 100`).
- Percentage columns use inline mini-bars built from Unicode block characters (e.g., `[blue]████[/]`) for visual scanning. No special Rich API required.
- Footer legend for the mini-bar colors (Matmul, ReLU, Overhead).

#### Verbose Footer

```
Use --verbose for full timing tables with raw times and per-layer breakdowns
```

### `--verbose` Output

When `--verbose` is passed, the default compact output is shown first, followed by the full current tables:
- Full Hardware Summary (multi-line, all fields)
- Skipped Backends with install hints
- Full Timing Results table (all columns including raw median, speedup, status)
- Full Primitive Breakdown table (including absolute seconds for matmul/relu)

This preserves all existing data for users who need it while keeping the default view clean.

### `--backends-help` Flag

New flag that prints install instructions for all known backends and exits immediately (no profiling). If no backends are skipped, prints "All backends are installed." This replaces having install hints in the default output.

### Dimension Formatting

For the compact `Dims` column, n_samples uses suffixes:
- < 1,000: show as-is (e.g., `64×4×500`)
- 1,000–999,999: `k` suffix (e.g., `64×4×10k`)
- >= 1,000,000: `M` suffix (e.g., `256×8×1M`)

## Files to Modify

- `src/network_estimation/profiler.py` — Replace `format_terminal_table()` with new `format_compact_output()` and `format_verbose_output()` functions. Rename existing `format_terminal_table()` to `format_verbose_output()` and keep it intact. Add `verbose: bool` parameter to `run_profile()` to control which formatter is called. The return type `(terminal_output, json_data)` stays the same.
- `src/network_estimation/cli.py` — Add `--verbose` and `--backends-help` flags to the `profile-simulation` subcommand argument parser. Pass `verbose` through to `run_profile()`. Handle `--backends-help` as an early exit before profiling.

## Terminal Width

Auto-detect terminal width via `shutil.get_terminal_size()` instead of the current hardcoded `width=140`. Fall back to 120 if detection fails.

## What Stays the Same

- All data collection (hardware, correctness, timing, breakdown) is unchanged.
- JSON output (`--output`) is unchanged.
- Progress bar during profiling is unchanged.
- The `--verbose` tables reuse the existing Rich table formatting code (no deletion, just gated behind the flag).
