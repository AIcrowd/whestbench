# Simplify GateDetailOverlay Wire Bands

The current overlay uses a complex split-window/focus-window system with context bars to show adjacent layer wires. This is confusing — users can't tell what the white gaps, split sections, and thin bars mean. Replace with a single continuous band showing all wires, with bold coral arrow indicators.

## Current Problems

- Split windows with `⋮` gap when inputs are far apart — looks like broken UI
- 3px context bars at top/bottom — too small to understand
- 7px black arrow triangles — nearly invisible
- `computeFocusWindow()` adds ~50 lines of complex branching logic for marginal benefit

## Design

### Continuous Full-Height Band

Each band renders **all N wires** as a single continuous vertical strip:

```
wireHeight = BAND_HEIGHT / N
```

For N=1024, each wire ≈ 0.27px — produces a smooth color gradient showing the layer's overall distribution. For small circuits (N=8), each wire gets ~35px. This scales naturally without any branching.

### Coral Arrow Indicators

Input/consumer wires are marked with **coral-colored triangles** (▶) pointing inward toward the band, placed at the wire's exact Y position:

```
              LEFT BAND                          RIGHT BAND
           ┌───────────┐                      ┌───────────┐
           │░░░░░░░░░░░│                      │░░░░░░░░░░░│
      ▶────│███████████│ ← input a (wire 104) │░░░░░░░░░░░│
           │░░░░░░░░░░░│                      │░░░░░░░░░░░│
           │░░░░░░░░░░░│                      │░░░░░░░░░░░│────◀ consumer
           │░░░░░░░░░░░│                      │░░░░░░░░░░░│
      ▶────│███████████│ ← input b (wire 928) │░░░░░░░░░░░│
           │░░░░░░░░░░░│                      │░░░░░░░░░░░│
           └───────────┘                      └───────────┘
              L107                               L109
```

Arrow specs:
- **Color**: `#F0524D` (coral, matches gate accent)
- **Size**: 10px tall triangles (up from 7px)
- **Position**: Outside the band edge, pointing inward
- **Highlighted wire**: Full-width stripe with 2px coral border (was black)

### Code Deleted

- `computeFocusWindow()` — entire function removed
- `drawContextBar()` — entire function removed
- Focus window logic in `drawBand()` — replaced with simple 0..N loop
- Split-window gap drawing — removed
- `leftFocus` / `rightFocus` memos — removed

## Files Changed

### [MODIFY] [GateDetailOverlay.jsx](file:///Users/mohanty/work/AIcrowd/challenges/alignment-research-center/circuit-estimation/circuit-estimation-mvp/tools/circuit-explorer/src/components/GateDetailOverlay.jsx)

- Delete `computeFocusWindow()` (lines 20-49)
- Delete `drawContextBar()` (lines 166-179)
- Rewrite `drawBand()` to iterate all N wires in one pass, compute `wireH = BAND_HEIGHT / n`, draw coral triangles at highlight positions
- Remove `leftFocus` / `rightFocus` useMemo hooks
- Simplify `drawBand` call sites (remove `focusWindow` param)

## Verification

### Manual Verification
1. Run `npm run dev` in `tools/circuit-explorer`
2. Hover over cells in the Circuit Heatmap
3. Verify bands show as continuous color gradients without gaps/splits/context bars
4. Verify coral triangles are clearly visible at input wire positions
5. Test with small circuit (width=8) — each wire should be a visible stripe
6. Test with large circuit (width=1024) — band should be a smooth gradient with arrows
