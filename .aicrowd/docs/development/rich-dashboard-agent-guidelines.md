# Rich Dashboard Agent Guidelines

> Reference for future agents working on the `cestim` terminal dashboard.
> Keep this doc updated when dashboard architecture or UX contracts change.

---

## 1) Non-Negotiable Output Contracts

These are hard constraints, not style preferences.

1. `--json` mode must emit JSON-only output on `stdout`.
- No Rich panels.
- No hints.
- No progress bars.
- No extra prose before or after JSON.

2. Human mode is append-only.
- Startup context prints first.
- Scoring progress prints next.
- Results sections append after scoring.
- Do not clear/redraw previously printed sections.

3. Human mode hints and dashboard content are written to `stdout`.
- Rich fallback errors may go to `stderr`.

4. `Run Context` and `Hardware & Runtime` must remain adjacent.
- They should render in the same row when width allows.

5. `Run Context` must show estimator identity at the top.
- `Estimator Class`
- `Estimator Path`

---

## 2) Where Things Live

- `src/circuit_estimation/cli.py`
- Owns mode routing (`--json` vs human), startup hints, progress wiring, and fallback behavior.
- Owns `--disable-rich` behavior (plain dashboard + classic tqdm).

- `src/circuit_estimation/reporting.py`
- Owns Rich panel/table/plot composition.
- Owns width/breakpoint logic and multi-pane arrangement.
- Should remain render-focused (avoid scoring-side logic).

- `src/circuit_estimation/scoring.py`
- Emits progress callbacks as budget x circuit work completes.
- Dashboard progress must be fed from this callback pipeline.

---

## 3) Rich vs rich-cli Notes

Important distinction:

- `rich` (Python library) is the implementation surface for this dashboard.
- `rich-cli` is mainly a rendering utility for files/markup in terminals.

For this project:

- Build layout with `Panel`, `Table`, `Align`, `Group`, and structured row helpers.
- Use `Live` + `Progress` for runtime progress updates.
- Prefer deterministic render trees over ad-hoc `Columns` chains.

Practical best-practice applied here:

1. Use explicit row builders (`_two_column_row`) for stable pane alignment.
2. Keep section order deterministic and append-only.
3. Keep progress in its own titled panel (`Scoring`) to reduce visual congestion.
4. Use width tiers to switch between stacked vs paired plot layouts.

---

## 4) Current Layout Strategy

### 4.1 Startup / Pre-run

1. Centered title panel.
2. Hint lines (`--json`, `--show-diagnostic-plots`, `--disable-rich`).
3. Context row:
- Left: `Run Context`
- Right: `Hardware & Runtime`
4. Scoring progress panel.

### 4.2 Post-run (append)

Always starts with:

1. `Scores`

Then:

- Without diagnostic plots:
- compact table-first sections.

- With diagnostic plots:
- `Budget` table first (full width), then plot-heavy rows.

### 4.3 Width Tiers (Wide Mode)

Breakpoint policy:

- `< 180`: `standard` mode (non-wide fallback flow).
- `180-199`: `compact` wide mode.
- `200-239`: `wide` mode.
- `>= 240`: `ultra` mode.

Behavior by tier:

1. `compact` (180-199)
- Plot sections are stacked vertically for readability.
- Avoid dense side-by-side plot pairs.

2. `wide` (200-239)
- Plot sections render in paired two-column rows.
- Higher information density while preserving legibility.

3. `ultra` (>=240)
- Same paired-row strategy as `wide`.
- Larger plot canvases for better signal detail.

---

## 5) Plot Rules

1. Plot panels should expand to fill their grid cells.
2. Use consistent border and legend styling across plots.
3. Keep chart legends external to plotted area to avoid overlap.
4. Sanitize plotext background ANSI escapes before rendering in panels.
5. Tune plot width from terminal tier, not from hardcoded constants alone.

Current max chart widths:

- `compact`: 84
- `wide`: 96
- `ultra`: 110
- fallback default: 76

---

## 6) Naming and Copy Guidelines

1. Avoid overloaded terms that conflict with CLI flags.
- Example applied: `Layer Diagnostics Histogram` renamed to `Layer MSE Histogram` to avoid confusion with `--show-diagnostic-plots`.

2. Keep panel titles short and semantically distinct.
- Good: `Scores`, `Budget`, `Profile Summary`, `Layer MSE Histogram`.

3. Prefer user-facing terms over internal jargon.

---

## 7) Change Safety Checklist (Required)

When changing dashboard layout or copy:

1. Run targeted tests first.
- `tests/test_reporting.py`
- `tests/test_cli.py`
- `tests/test_cli_fallback.py`

2. Verify these classes of behavior:
- JSON-only output contract.
- Append-only human flow.
- Context/hardware adjacency.
- Breakpoint-specific layout behavior.
- Profile + diagnostic-plot combinations.

3. Run full quality gates:
- `uv run ruff check src tests`
- `uv run pyright`
- `uv run pytest -q`

4. Manual smoke in representative widths:
- `COLUMNS=180`
- `COLUMNS=200`
- `COLUMNS=240`

Example commands:

```bash
uv run cestim --json | jq .
COLUMNS=180 uv run cestim --profile --show-diagnostic-plots
COLUMNS=200 uv run cestim --profile --show-diagnostic-plots
COLUMNS=240 uv run cestim --profile --show-diagnostic-plots
uv run cestim --disable-rich
```

---

## 8) Common Pitfalls

1. Reintroducing non-JSON text in `--json` mode.
2. Using free-form `Columns` chains that produce uneven "masonry" layouts.
3. Mixing pre-run hint text into post-run dashboard sections.
4. Printing absolute filesystem paths when user-input path is expected for estimator path display.
5. Tweaking visuals without adding/updating tests that lock in behavior.

---

## 9) Definition of Done for Dashboard Changes

A dashboard change is not complete unless:

1. Contracts in section 1 still hold.
2. Breakpoint behavior is intentional and tested.
3. Full lint/type/test suite passes.
4. Human-mode output has been manually checked in at least one narrow and one wide terminal.
5. This guideline doc is updated if behavior or policy changed.

