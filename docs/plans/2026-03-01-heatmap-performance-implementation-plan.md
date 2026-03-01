# Heatmap Performance Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Make the Circuit Explorer performant and usable at 1024×256 circuits in heatmap mode.

**Architecture:** Web Worker for computation, canvas for all dense rendering, debounced hover with focus-window overlay, conditional panel hiding. No new dependencies.

**Tech Stack:** Vite (existing), Web Workers (native), Canvas 2D API (native), React hooks

**Design doc:** [`2026-03-01-heatmap-performance-design.md`](file:///Users/mohanty/work/AIcrowd/challenges/alignment-research-center/circuit-estimation/circuit-estimation-mvp/docs/plans/2026-03-01-heatmap-performance-design.md)

---

### Task 1: Web Worker for Computation

Move `empiricalMean` and `meanPropagation` off the main thread.

**Files:**
- Create: `tools/circuit-explorer/src/circuit.worker.js`
- Create: `tools/circuit-explorer/src/useWorker.js`
- Modify: `tools/circuit-explorer/src/components/EstimatorRunner.jsx`
- Modify: `tools/circuit-explorer/src/App.jsx` (tour auto-run effects)

**Step 1: Create the worker file**

`circuit.worker.js` imports `circuit.js` and `estimators.js`, listens for messages, runs the requested computation, and posts results back. Vite supports `new Worker(new URL('./circuit.worker.js', import.meta.url), { type: 'module' })` natively.

```js
// circuit.worker.js
import { empiricalMean, randomCircuit } from './circuit';
import { meanPropagation } from './estimators';

self.onmessage = function (e) {
  const { id, type, params } = e.data;

  // Reconstruct circuit from plain object (typed arrays survive postMessage)
  const circuit = params.circuit;

  let result;
  const t0 = performance.now();

  switch (type) {
    case 'empiricalMean': {
      result = { estimates: empiricalMean(circuit, params.trials, params.seed) };
      break;
    }
    case 'meanPropagation': {
      result = { estimates: meanPropagation(circuit) };
      break;
    }
    default:
      self.postMessage({ id, error: `Unknown type: ${type}` });
      return;
  }

  result.time = performance.now() - t0;
  self.postMessage({ id, result });
};
```

**Step 2: Create `useWorker` hook**

A React hook that manages the worker lifecycle: creates the worker once, provides a `run(type, params)` function that returns a promise, and tracks `isRunning` state.

```js
// useWorker.js
import { useCallback, useEffect, useRef, useState } from 'react';

export function useCircuitWorker() {
  const workerRef = useRef(null);
  const callbackRef = useRef(null);
  const idRef = useRef(0);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    const worker = new Worker(
      new URL('./circuit.worker.js', import.meta.url),
      { type: 'module' }
    );
    worker.onmessage = (e) => {
      const { id, result, error } = e.data;
      if (callbackRef.current?.id === id) {
        if (error) callbackRef.current.reject(new Error(error));
        else callbackRef.current.resolve(result);
        callbackRef.current = null;
        setIsRunning(false);
      }
    };
    workerRef.current = worker;
    return () => worker.terminate();
  }, []);

  const run = useCallback((type, params) => {
    return new Promise((resolve, reject) => {
      const id = ++idRef.current;
      callbackRef.current = { id, resolve, reject };
      setIsRunning(true);
      workerRef.current.postMessage({ id, type, params });
    });
  }, []);

  return { run, isRunning };
}
```

**Step 3: Update `EstimatorRunner` to use the worker**

Replace synchronous `empiricalMean()` / `meanPropagation()` calls with `worker.run(...)`. The `useCircuitWorker` hook is instantiated in `App.jsx` and passed as a prop, so a single worker is shared.

Key changes in `EstimatorRunner.jsx`:
- Accept `worker` prop (the `{ run, isRunning }` object)
- `runGroundTruth`: `const result = await worker.run('empiricalMean', { circuit, trials: 10000, seed: 7777 })`
- `runSampling`: `const result = await worker.run('empiricalMean', { circuit, trials: budget, seed: 1234 })`
- `runMeanProp`: `const result = await worker.run('meanPropagation', { circuit })`
- Remove `requestAnimationFrame` wrapper (no longer needed — computation is off-thread)
- `running` state tracks which estimator the worker is processing

**Step 4: Update `App.jsx` tour auto-run effects**

The tour auto-run effects (steps 3, 4, 5) currently call `empiricalMean` synchronously. Update them to use the worker:

```js
// In App.jsx
const worker = useCircuitWorker();

// Step 3: auto-run ground truth
useEffect(() => {
  if (step === 3 && !autoRunDone.current.gt) {
    autoRunDone.current.gt = true;
    worker.run('empiricalMean', { circuit: tourCircuit, trials: 10000, seed: 99 })
      .then(({ estimates, time }) => {
        setTourGroundTruth(estimates);
        setTourGroundTruthTime(time);
      });
  }
}, [step, tourCircuit, worker]);
```

Same pattern for steps 4 (sampling) and 5 (meanPropagation).

**Step 5: Verify and commit**

Run: `cd tools/circuit-explorer && npm run dev`
- Open browser, go through tour steps 3–5 — UI should remain responsive
- Switch to explore mode, set width=1024 depth=256, run Ground Truth — browser should NOT freeze
- Verify timing badge shows correct elapsed time

```bash
git add tools/circuit-explorer/src/circuit.worker.js tools/circuit-explorer/src/useWorker.js
git add tools/circuit-explorer/src/components/EstimatorRunner.jsx tools/circuit-explorer/src/App.jsx
git commit -m "perf: move computation to Web Worker — unblock main thread"
```

---

### Task 2: Fix Heatmap Canvas Overflow

The heatmap canvas overflows its parent `.panel` container at large wire counts.

**Files:**
- Modify: `tools/circuit-explorer/src/components/CircuitHeatmap.jsx`
- Modify: `tools/circuit-explorer/src/App.css` (add overflow constraint)

**Step 1: Fix canvas sizing**

In `CircuitHeatmap.jsx`, the height is `Math.max(300, Math.min(600, n * 3))`. At 1024 wires this gives 600px, but the canvas width isn't properly constrained to the container either. Fix:

- Set container to `overflow: hidden` in CSS
- Use `container.getBoundingClientRect().width` for width (already done, but ensure the canvas `style.width` doesn't exceed it)
- Cap height to a sensible max: `Math.min(500, containerHeight)` where `containerHeight` is derived from the viewport
- Add `box-sizing: border-box` to the canvas

CSS fix in `App.css`:
```css
.circuit-heatmap {
  overflow: hidden;
}
.circuit-heatmap canvas {
  max-width: 100%;
  display: block;
}
```

**Step 2: Add crosshair overlay canvas**

Add a second canvas positioned `absolute` on top of the heatmap canvas, same dimensions. This canvas is cleared and redrawn on mousemove (just two lines — instant), avoiding full heatmap redraw.

**Step 3: Verify and commit**

Run: `npm run dev`
- Set width=1024, depth=256 → heatmap should be fully contained within its panel
- Hover over the heatmap → crosshair should follow smoothly without redrawing the heat grid

```bash
git add tools/circuit-explorer/src/components/CircuitHeatmap.jsx tools/circuit-explorer/src/App.css
git commit -m "fix: constrain heatmap canvas to panel, add crosshair overlay"
```

---

### Task 3: Hide SignalHeatmap in Heatmap Mode

**Files:**
- Modify: `tools/circuit-explorer/src/App.jsx`

**Step 1: Conditionally render SignalHeatmap**

In `App.jsx`, the `SignalHeatmap` is rendered inside the explore-mode panels section. Wrap it in `useGraphMode` check:

```jsx
{/* Only show SignalHeatmap in graph mode — in heatmap mode, CircuitHeatmap already shows this data */}
{useGraphMode && (
  <SignalHeatmap
    means={exploreDisplayMeans}
    width={params.width}
    depth={params.depth}
  />
)}
```

The `WireStats` component stays — it shows mean ± σ as a line chart (different from the heatmap grid).

**Step 2: Verify and commit**

Run: `npm run dev`
- Set width=8, depth=6 → SignalHeatmap should appear below graph
- Set width=1024, depth=256 → SignalHeatmap should NOT appear (heatmap mode)
- WireStats should appear in both modes

```bash
git add tools/circuit-explorer/src/App.jsx
git commit -m "perf: hide redundant SignalHeatmap in heatmap mode"
```

---

### Task 4: GateDetailOverlay — Canvas Bands + Focus Window + Debounce

**Files:**
- Modify: `tools/circuit-explorer/src/components/GateDetailOverlay.jsx`
- Modify: `tools/circuit-explorer/src/components/CircuitHeatmap.jsx` (debounce)

**Step 1: Debounce hover in CircuitHeatmap**

Replace raw `onMouseMove` with RAF-gated handler that only updates state when the hovered **cell** changes:

```js
const lastCellRef = useRef(null);
const rafRef = useRef(null);

const handleMouseMove = useCallback((e) => {
  if (rafRef.current) return; // already scheduled
  rafRef.current = requestAnimationFrame(() => {
    rafRef.current = null;
    if (!dims.cellW) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const layer = Math.floor((e.clientX - rect.left) / dims.cellW);
    const wire = Math.floor((e.clientY - rect.top) / dims.cellH);
    const cellKey = `${layer},${wire}`;
    if (cellKey === lastCellRef.current) return; // same cell, skip
    lastCellRef.current = cellKey;
    if (layer >= 0 && layer < d && wire >= 0 && wire < n) {
      setHovered({ wire, layer, x: e.clientX - containerRef.current.getBoundingClientRect().left, y: e.clientY - containerRef.current.getBoundingClientRect().top });
    } else {
      setHovered(null);
    }
  });
}, [dims, n, d]);
```

**Step 2: Replace SVG bands with canvas in GateDetailOverlay**

Replace the left/right band `<svg>` elements with `<canvas>` elements drawn via `useEffect`. Each band is 24×280px max. Drawing 1024 colored rects to canvas takes ~0.1ms.

**Step 3: Implement focus window for large wire counts**

When `n > 100`, instead of showing all wires in the band:
- Determine a "window" of ~60 wires centered on the input wires (`first[w]` and `second[w]`)
- If inputs are far apart (gap > 40), show two mini-windows with a gap indicator
- Top/bottom of band show a 3px compressed summary gradient
- Wire index labels appear next to input/output wires

```js
function computeFocusWindow(n, inputFirst, inputSecond, maxVisible = 60) {
  if (n <= maxVisible) {
    return { type: 'full', start: 0, end: n }; // show all
  }
  const lo = Math.min(inputFirst, inputSecond);
  const hi = Math.max(inputFirst, inputSecond);
  const gap = hi - lo;

  if (gap <= maxVisible - 10) {
    // Inputs close enough — single window centered on both
    const center = Math.floor((lo + hi) / 2);
    const half = Math.floor(maxVisible / 2);
    const start = Math.max(0, Math.min(n - maxVisible, center - half));
    return { type: 'single', start, end: start + maxVisible };
  } else {
    // Inputs far apart — two mini-windows
    const halfWin = Math.floor(maxVisible / 2) - 2; // leave room for gap indicator
    return {
      type: 'split',
      windowA: { start: Math.max(0, lo - Math.floor(halfWin / 2)), end: Math.min(n, lo + Math.ceil(halfWin / 2)) },
      windowB: { start: Math.max(0, hi - Math.floor(halfWin / 2)), end: Math.min(n, hi + Math.ceil(halfWin / 2)) },
    };
  }
}
```

**Step 4: Verify and commit**

Run: `npm run dev`
- Set width=1024, depth=256, run Ground Truth
- Hover over heatmap cells — overlay should appear instantly without jank
- Overlay should show compact focus window, not 1024px tall bands
- Input wires should be highlighted with ▶ markers and labeled
- Verify overlay stays within viewport bounds

```bash
git add tools/circuit-explorer/src/components/GateDetailOverlay.jsx
git add tools/circuit-explorer/src/components/CircuitHeatmap.jsx
git commit -m "perf: canvas bands + focus window + debounce on GateDetailOverlay"
```

---

### Task 5: Hybrid Layer Linking (activeLayer from Heatmap)

**Files:**
- Modify: `tools/circuit-explorer/src/components/CircuitHeatmap.jsx` (click handler)
- Modify: `tools/circuit-explorer/src/App.jsx` (pass activeLayer + onLayerClick)
- Modify: `tools/circuit-explorer/src/components/GateStats.jsx` (highlight active layer)
- Modify: `tools/circuit-explorer/src/components/WireStats.jsx` (reference line)
- Modify: `tools/circuit-explorer/src/components/EstimatorComparison.jsx` (highlight bar)

**Step 1: Add click handler to CircuitHeatmap**

Clicking a cell sets `activeLayer` to that cell's layer index. Clicking the same layer again clears it. Pass `onLayerClick` and `activeLayer` as props.

```jsx
// In CircuitHeatmap.jsx
const handleClick = useCallback((e) => {
  if (!dims.cellW) return;
  const rect = canvasRef.current.getBoundingClientRect();
  const layer = Math.floor((e.clientX - rect.left) / dims.cellW);
  if (layer >= 0 && layer < d) {
    onLayerClick?.(layer === activeLayer ? undefined : layer);
  }
}, [dims, d, activeLayer, onLayerClick]);
```

Draw a subtle column highlight on the crosshair overlay canvas when `activeLayer` is set.

**Step 2: Highlight activeLayer in analytics panels**

- `GateStats`: Active layer bar gets full opacity, others dim to 30%
- `WireStats`: Vertical `ReferenceLine` at `activeLayer`
- `EstimatorComparison`: Active layer bar highlighted

These are small prop additions — the panels already receive layer data indexed by position.

**Step 3: Add Escape key handler**

In `App.jsx`, add a `keydown` listener for Escape that clears `activeLayer`:

```js
useEffect(() => {
  const handler = (e) => {
    if (e.key === 'Escape') setActiveLayer(undefined);
  };
  window.addEventListener('keydown', handler);
  return () => window.removeEventListener('keydown', handler);
}, []);
```

**Step 4: Verify and commit**

Run: `npm run dev`
- Set width=128, depth=64, run Ground Truth
- Click a layer column on heatmap → GateStats, WireStats, and EstimatorComparison should highlight that layer
- Click same column again → highlights clear
- Press Escape → highlights clear

```bash
git add tools/circuit-explorer/src/components/CircuitHeatmap.jsx
git add tools/circuit-explorer/src/App.jsx
git add tools/circuit-explorer/src/components/GateStats.jsx
git add tools/circuit-explorer/src/components/WireStats.jsx
git add tools/circuit-explorer/src/components/EstimatorComparison.jsx
git commit -m "feat: hybrid layer linking — click heatmap to highlight across panels"
```

---

## Verification Plan

### Browser Testing (manual, via dev server)

All verification done via `cd tools/circuit-explorer && npm run dev` and opening the browser.

1. **Small circuit (8×6)**: Tour steps 3–5 should still work. SignalHeatmap visible. UI remains responsive.
2. **Medium circuit (64×32)**: Graph mode. Run all three estimators. Verify no UI freezes.
3. **Threshold circuit (65×64)**: Should auto-switch to heatmap mode. SignalHeatmap hidden.
4. **Large circuit (1024×256)**: Heatmap mode. Run Ground Truth — browser should NOT freeze. Heatmap renders within panel bounds. Hover overlay is responsive. Focus window shows ~60 wires around inputs. Click layer → analytics panels highlight.
5. **Layer linking**: Click layer on heatmap → check GateStats, WireStats, EstimatorComparison. Escape clears.

### Build Verification

```bash
cd tools/circuit-explorer && npm run build
```

Should complete without errors or warnings.
