# Large Circuit Heatmap Performance — Design

## Problem

The Circuit Explorer chokes at 1024×256 circuits (262k gates). Four bottlenecks compound:

1. **Computation**: `empiricalMean` does ~2.6B ops synchronously on the main thread → browser hangs for seconds
2. **SignalHeatmap SVG**: Creates 262k `<rect>` React nodes → React reconciliation collapses
3. **GateDetailOverlay**: Renders 1024 SVG elements per band on every hover → visible jank
4. **GateDetailOverlay height**: At 1024 wires, bands are ~1024px tall → overlay is unusable

## Approach: Web Worker + All-Canvas + YAGNI

### 1. Web Worker for Computation

A single `circuit.worker.js` that mirrors the existing `circuit.js` + `estimators.js` API.

```
Main thread                          Worker thread
─────────────                        ─────────────
postMessage({                   →    Receives job
  type: 'empiricalMean',             Runs empiricalMean()
  circuit, trials, seed              (same code, no UI freezing)
})
                                ←    postMessage({ estimates, time })
onmessage → setEstimatorResults()
```

- The circuit object (typed arrays) is **transferable** — zero-copy to the worker
- Results (`Float32Array` per layer) are also transferable back
- A `useWorker()` hook wraps the `postMessage`/`onmessage` pattern, returning `{ run, result, isRunning }`
- Worker runs `empiricalMean`, `meanPropagation`, and any future estimators
- `EstimatorRunner` calls `worker.run('groundTruth', { circuit, trials: 10000 })` instead of computing inline
- Progress: worker posts intermediate progress (every 1000 samples) for a real progress bar

`circuit.js` and `estimators.js` stay unchanged — the worker just imports them.

### 2. Hide SignalHeatmap in Heatmap Mode

`SignalHeatmap` (SVG, 262k rects) and `CircuitHeatmap` (canvas) show the same data (wire means as a color grid) with transposed axes. In heatmap mode, `SignalHeatmap` is a redundant, slower duplicate.

Fix: `{useGraphMode && <SignalHeatmap ... />}` — just don't render it.

This eliminates the worst rendering bottleneck entirely.

### 3. CircuitHeatmap Canvas Optimization

The existing canvas renderer is mostly fine for 262k `fillRect` calls. Two tweaks:

- **Resolution cap**: If cells are sub-pixel (1024 wires → each cell < 0.5px tall), draw at reduced resolution and let the browser scale the canvas. ~4× faster draw for free.
- **Crosshair overlay canvas**: A second canvas layer on top for the hover crosshair (horizontal + vertical lines). Avoids redrawing the entire heatmap on every mouse move.

### 4. GateDetailOverlay — Canvas Bands + Focus Window

**Canvas bands** replace SVG:
- Left/right band SVGs become two tiny canvases (24×280px)
- Drawing 1024 colored rectangles to canvas is ~0.1ms
- Input/output wires drawn as a second pass with accent strokes

**Mouse debouncing**:
- `requestAnimationFrame`-gated handler — one update per frame (16ms max)
- Only update overlay when the hovered **cell** changes, not on every mouse pixel

**Focus window** for tall circuits — instead of showing all `n` wires:
- Show ±30 wires around the input wires (both `first[w]` and `second[w]`)
- Max band height: **280px** (the existing `BAND_HEIGHT` constant)
- If both inputs are far apart (e.g. wire 29 and wire 780), show **two mini-windows** with a gap indicator
- Top/bottom of band show a 3px compressed summary bar (mini-gradient of all wires above/below the window)
- Wire index labels appear next to the input/output wires for orientation

```
┌──────┐   ┌─────┐   ┌──────┐
│ L92  │   │Gate │   │ L94  │
│ ░░░  │   │     │   │ ░░░  │  ← compressed context (wires above)
│ ███  │   │ c+  │   │      │
│ ▶██  │───│ax + │───│ ██◀  │  ← input/output wires + ~30 neighbors
│ ▶██  │   │by + │   │ ██◀  │
│ ███  │   │pxy  │   │      │
│ ░░░  │   │     │   │ ░░░  │  ← compressed context (wires below)
└──────┘   └─────┘   └──────┘
```

### 5. Dashboard Layout in Heatmap Mode

```
┌──────────────────────────────────────────┐
│ Header + Step Indicator                  │
├─────────┬────────────────────────────────┤
│ Sidebar │  CircuitHeatmap (canvas)       │
│ - Params│  1024×256 color grid           │
│ - Run   │  + crosshair overlay           │
│ - Budget│  + GateDetailOverlay on hover  │
│         ├────────────────────────────────┤
│         │ Gate Stats (per-layer bars)    │
│         ├────────────────────────────────┤
│         │ WireStats (mean ± σ per layer) │
│         ├────────────────────────────────┤
│         │ EstimatorComparison (MSE bars) │
└─────────┴────────────────────────────────┘
```

- `CircuitHeatmap` replaces `CircuitGraphJoint` (already implemented)
- `SignalHeatmap` hidden (redundant)
- `GateStats`, `WireStats`, `EstimatorComparison` unchanged — they aggregate per-layer, 256 data points is trivial

**Hybrid layer linking:**
- Clicking a column on `CircuitHeatmap` sets `activeLayer` in App state
- `GateStats` highlights the clicked layer's bar (others dim to 30%)
- `WireStats` shows a vertical reference line at that layer
- `EstimatorComparison` highlights that layer's MSE bar
- Click again or Escape to clear. Reuses existing `activeLayer` state.

## What's Deferred (YAGNI / Future)

- **WebGPU compute shaders** — would give 50-100× speedup for sampling, but adds complexity and Safari doesn't support it. Ship as progressive enhancement behind `?gpu=1` flag later.
- **Brush/region selection** on heatmap
- **Zoom/pan** on heatmap (hover overlay gives detail)
- **Side-by-side heatmap comparison** (GT vs estimator) — MSE panel serves this aggregate purpose
