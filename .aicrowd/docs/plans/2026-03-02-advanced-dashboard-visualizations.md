# Advanced Dashboard Visualizations Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Add 5 new visualization components to the Circuit Explorer: coefficient histograms, activation ribbon, std heatmap, mean-prop error heatmap, and animated signal flow.

**Architecture:** Enrich the data pipeline (`empiricalStats`) to compute std/min/max alongside means. Build 5 new React components using Canvas rendering (no extra deps). Thread data from `App.jsx` down to new panels in Explore mode.

**Tech Stack:** React 19, Canvas 2D, tfjs-vis (existing), Vite

---

### Task 1: Enrich Data Pipeline — `empiricalStats` (CPU)

**Files:**
- Modify: `tools/circuit-explorer/src/circuit.js`
- Modify: `tools/circuit-explorer/src/circuit.worker.js`

**Step 1: Add `empiricalStats` function to `circuit.js`**

Add after the existing `empiricalMean` function (~line 161):

```js
/**
 * Run circuit on random ±1 inputs and return rich stats per wire per layer.
 * Returns: { means, stds, mins, maxs } — each Float32Array[] (one per layer, length n)
 */
export function empiricalStats(circuit, trials, seed = 99) {
  const rng = makeRng(seed);
  const inputs = [];
  for (let t = 0; t < trials; t++) {
    const row = new Float32Array(circuit.n);
    for (let i = 0; i < circuit.n; i++) {
      row[i] = rng.randBool() ? 1.0 : -1.0;
    }
    inputs.push(row);
  }

  const layerOutputs = runBatched(circuit, inputs);
  const means = [];
  const stds = [];
  const mins = [];
  const maxs = [];

  for (const batch of layerOutputs) {
    const n = circuit.n;
    const mean = new Float32Array(n);
    const sumSq = new Float32Array(n);
    const min = new Float32Array(n).fill(Infinity);
    const max = new Float32Array(n).fill(-Infinity);

    for (let b = 0; b < trials; b++) {
      for (let i = 0; i < n; i++) {
        const v = batch[b][i];
        mean[i] += v;
        sumSq[i] += v * v;
        if (v < min[i]) min[i] = v;
        if (v > max[i]) max[i] = v;
      }
    }

    const std = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      mean[i] /= trials;
      const variance = sumSq[i] / trials - mean[i] * mean[i];
      std[i] = Math.sqrt(Math.max(0, variance));
    }

    means.push(mean);
    stds.push(std);
    mins.push(min);
    maxs.push(max);
  }

  return { means, stds, mins, maxs };
}
```

**Step 2: Add `runSingleTrial` convenience function to `circuit.js`**

Add after `empiricalStats`:

```js
/**
 * Run circuit with a single random ±1 input.
 * Returns: Float32Array[] — one per layer, each of length n (the wire values at that layer).
 */
export function runSingleTrial(circuit, seed = 42) {
  const rng = makeRng(seed);
  const input = new Float32Array(circuit.n);
  for (let i = 0; i < circuit.n; i++) {
    input[i] = rng.randBool() ? 1.0 : -1.0;
  }
  const results = runBatched(circuit, [input]);
  return results.map(batch => batch[0]);
}
```

**Step 3: Add worker handlers in `circuit.worker.js`**

Add new cases to the switch statement:

```js
case 'empiricalStats': {
  const circuit = params.circuit;
  const stats = empiricalStats(circuit, params.trials, params.seed);
  result = { estimates: stats.means, stds: stats.stds, mins: stats.mins, maxs: stats.maxs };
  break;
}
case 'runSingleTrial': {
  const circuit = params.circuit;
  result = { layerValues: runSingleTrial(circuit, params.seed) };
  break;
}
```

Update the import at top of worker file:
```js
import { empiricalMean, empiricalStats, randomCircuit, runSingleTrial } from './circuit';
```

**Step 4: Verify — start dev server and check console**

Run: `cd tools/circuit-explorer && npm run dev`
Open browser, open console, check for no import errors.

**Step 5: Commit**

```bash
git add tools/circuit-explorer/src/circuit.js tools/circuit-explorer/src/circuit.worker.js
git commit -m "feat: add empiricalStats and runSingleTrial to data pipeline"
```

---

### Task 2: Enrich Data Pipeline — `empiricalStatsTF` (GPU)

**Files:**
- Modify: `tools/circuit-explorer/src/circuit-tf.js`

**Step 1: Add `empiricalStatsTF` function**

Add after the existing `empiricalMeanTF` function. This computes mean, std, min, max per layer using TF.js tensor ops:

```js
/**
 * GPU-accelerated empirical stats (mean, std, min, max per wire per layer).
 * Returns: { means, stds, mins, maxs } — each Float32Array[], one per layer, length n.
 */
export async function empiricalStatsTF(circuit, trials, seed = 99, onProgress = null) {
  await initTF();

  const rng = makeRng(seed);
  const inputData = new Float32Array(trials * circuit.n);
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = rng.random() < 0.5 ? -1.0 : 1.0;
  }

  let x = tf.tensor2d(inputData, [trials, circuit.n]);
  const means = [], stds = [], mins = [], maxs = [];
  const totalLayers = circuit.gates.length;

  for (let li = 0; li < totalLayers; li++) {
    const layer = circuit.gates[li];
    const n = circuit.n;
    const firstArr = new Array(n), secondArr = new Array(n);
    const constArr = new Array(n), aArr = new Array(n);
    const bArr = new Array(n), pArr = new Array(n);

    for (let i = 0; i < n; i++) {
      firstArr[i] = layer.first[i];
      secondArr[i] = layer.second[i];
      constArr[i] = layer['const'][i];
      aArr[i] = layer.firstCoeff[i];
      bArr[i] = layer.secondCoeff[i];
      pArr[i] = layer.productCoeff[i];
    }

    const newX = tf.tidy(() => {
      const idx1 = tf.tensor1d(firstArr, 'int32');
      const idx2 = tf.tensor1d(secondArr, 'int32');
      const c = tf.tensor1d(constArr);
      const a = tf.tensor1d(aArr);
      const b = tf.tensor1d(bArr);
      const p = tf.tensor1d(pArr);
      const xf = tf.gather(x, idx1, 1);
      const xs = tf.gather(x, idx2, 1);
      const xfxs = tf.mul(xf, xs);
      return tf.add(
        tf.add(c, tf.mul(a, xf)),
        tf.add(tf.mul(b, xs), tf.mul(p, xfxs))
      );
    });

    // Compute stats and pull to CPU
    const [meanT, varianceT] = tf.moments(newX, 0);
    const minT = tf.min(newX, 0);
    const maxT = tf.max(newX, 0);
    const stdT = tf.sqrt(varianceT);

    const [meanData, stdData, minData, maxData] = await Promise.all([
      meanT.data(), stdT.data(), minT.data(), maxT.data()
    ]);

    means.push(Float32Array.from(meanData));
    stds.push(Float32Array.from(stdData));
    mins.push(Float32Array.from(minData));
    maxs.push(Float32Array.from(maxData));

    meanT.dispose(); varianceT.dispose(); stdT.dispose();
    minT.dispose(); maxT.dispose();
    x.dispose();
    x = newX;

    if (onProgress) onProgress((li + 1) / totalLayers);
  }

  x.dispose();
  return { means, stds, mins, maxs };
}
```

**Step 2: Export the new function**

Ensure `empiricalStatsTF` is exported (it uses `export async function` so it's already exported).

**Step 3: Commit**

```bash
git add tools/circuit-explorer/src/circuit-tf.js
git commit -m "feat: add empiricalStatsTF for GPU-accelerated stats"
```

---

### Task 3: Wire Stats Through EstimatorRunner and App

**Files:**
- Modify: `tools/circuit-explorer/src/components/EstimatorRunner.jsx`
- Modify: `tools/circuit-explorer/src/App.jsx`

**Step 1: Update EstimatorRunner to use `empiricalStats`**

In `EstimatorRunner.jsx`, import the new TF function:
```js
import { empiricalMeanTF, empiricalStatsTF, initTF } from "../circuit-tf";
```

Modify `runEmpirical` to use `empiricalStatsTF` for ground truth runs. The key change: when `key === 'groundTruth'`, use `empiricalStatsTF` to get richer data. For sampling, continue using `empiricalMeanTF` (faster, stats not needed for comparison).

In the GPU path of `runEmpirical`:
```js
if (key === 'groundTruth' && tfBackend && tfBackend !== 'unavailable') {
  // Use stats version for ground truth
  perfStart(`estimator-${key}`);
  const t0 = performance.now();
  const stats = await empiricalStatsTF(circuit, trials, seed, (p) => setProgress(p));
  time = performance.now() - t0;
  perfEnd(`estimator-${key}`);
  estimates = stats.means;
  extraStats = { stds: stats.stds, mins: stats.mins, maxs: stats.maxs };
}
```

Add `extraStats` to the enriched result:
```js
const enriched = { ...displayInfo, estimates, time, ...(extraStats || {}) };
```

For the CPU fallback, update the worker call to use `'empiricalStats'`:
```js
const result = await worker.run('empiricalStats', { circuit, trials, seed });
estimates = result.estimates;
extraStats = { stds: result.stds, mins: result.mins, maxs: result.maxs };
```

**Step 2: Thread stats through App.jsx**

Add derived state for stats:
```js
const groundTruthStats = estimatorResults.groundTruth || null;
```

The `estimatorResults.groundTruth` object will now contain `stds`, `mins`, `maxs` alongside `estimates`. Pass these to new components (added in later tasks).

**Step 3: Verify — run dev server, click Ground Truth button**

Run: `cd tools/circuit-explorer && npm run dev`
Open browser, skip tour to Explore mode, click "Ground Truth (10k)". Check console for no errors. Verify the circuit still colors correctly.

**Step 4: Commit**

```bash
git add tools/circuit-explorer/src/components/EstimatorRunner.jsx tools/circuit-explorer/src/App.jsx
git commit -m "feat: thread enriched stats through EstimatorRunner and App"
```

---

### Task 4: Coefficient Histograms Component

**Files:**
- Create: `tools/circuit-explorer/src/components/CoeffHistograms.jsx`
- Modify: `tools/circuit-explorer/src/App.jsx`
- Modify: `tools/circuit-explorer/src/App.css`

**Step 1: Create `CoeffHistograms.jsx`**

Canvas-based component. For each layer, draws 4 mini-histograms (c, a, b, p) showing coefficient value distributions across all wires.

```jsx
/**
 * CoeffHistograms — Layer-by-layer coefficient distribution sparklines.
 * For each layer, shows 4 mini bar-histograms (c, a, b, p coefficients).
 * Canvas-rendered for performance with large circuits.
 */
import { useEffect, useRef, useMemo } from "react";

const COLORS = {
  c: "#D1D5DB", a: "#94A3B8", b: "#334155", p: "#F0524D",
};
const COEFF_KEYS = ["c", "a", "b", "p"];
const COEFF_LABELS = {
  c: "bias (c)", a: "first (a)", b: "second (b)", p: "product (p)",
};
const NUM_BINS = 12;

function binValues(values, numBins = NUM_BINS) {
  if (values.length === 0) return { bins: new Array(numBins).fill(0), min: -1, max: 1 };
  let min = Infinity, max = -Infinity;
  for (const v of values) { if (v < min) min = v; if (v > max) max = v; }
  if (min === max) { min -= 0.5; max += 0.5; }
  const binWidth = (max - min) / numBins;
  const bins = new Array(numBins).fill(0);
  for (const v of values) {
    const idx = Math.min(numBins - 1, Math.floor((v - min) / binWidth));
    bins[idx]++;
  }
  return { bins, min, max };
}

export default function CoeffHistograms({ circuit }) {
  const canvasRef = useRef(null);

  const layerHistos = useMemo(() => {
    if (!circuit) return [];
    const { n, d, gates } = circuit;
    const result = [];
    for (let l = 0; l < d; l++) {
      const layer = gates[l];
      const coeffs = {
        c: [], a: [], b: [], p: [],
      };
      for (let i = 0; i < n; i++) {
        coeffs.c.push(layer.const[i]);
        coeffs.a.push(layer.firstCoeff[i]);
        coeffs.b.push(layer.secondCoeff[i]);
        coeffs.p.push(layer.productCoeff[i]);
      }
      result.push({
        c: binValues(coeffs.c),
        a: binValues(coeffs.a),
        b: binValues(coeffs.b),
        p: binValues(coeffs.p),
      });
    }
    return result;
  }, [circuit]);

  useEffect(() => {
    if (!canvasRef.current || layerHistos.length === 0) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const containerW = container.offsetWidth || 800;

    const d = layerHistos.length;
    const LAYER_W = Math.max(48, Math.min(80, Math.floor((containerW - 40) / d)));
    const HIST_H = 28;
    const GAP = 2;
    const LAYER_GAP = 4;
    const LABEL_H = 12;
    const totalH = COEFF_KEYS.length * (HIST_H + GAP) + LABEL_H + 8;
    const totalW = d * (LAYER_W + LAYER_GAP) + 40;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = totalW * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = `${totalW}px`;
    canvas.style.height = `${totalH}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, totalW, totalH);

    for (let l = 0; l < d; l++) {
      const x0 = 36 + l * (LAYER_W + LAYER_GAP);
      const histo = layerHistos[l];

      COEFF_KEYS.forEach((key, ki) => {
        const y0 = ki * (HIST_H + GAP);
        const { bins } = histo[key];
        const maxBin = Math.max(1, ...bins);
        const barW = LAYER_W / bins.length;

        for (let bi = 0; bi < bins.length; bi++) {
          const barH = (bins[bi] / maxBin) * HIST_H;
          ctx.fillStyle = COLORS[key];
          ctx.globalAlpha = 0.3 + 0.7 * (bins[bi] / maxBin);
          ctx.fillRect(x0 + bi * barW, y0 + HIST_H - barH, barW - 0.5, barH);
        }
        ctx.globalAlpha = 1;
      });

      // Layer label
      ctx.fillStyle = "#9CA3AF";
      ctx.font = "9px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(`L${l}`, x0 + LAYER_W / 2, totalH - 2);
    }

    // Y-axis labels
    ctx.textAlign = "right";
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "8px 'IBM Plex Mono', monospace";
    COEFF_KEYS.forEach((key, ki) => {
      const y = ki * (HIST_H + GAP) + HIST_H / 2 + 3;
      ctx.fillText(key, 30, y);
    });
  }, [layerHistos]);

  if (!circuit) return null;

  return (
    <div className="panel">
      <h2>Coefficient Distributions</h2>
      <div className="formula-legend" style={{ marginBottom: 8 }}>
        {COEFF_KEYS.map(k => (
          <span key={k} style={{ color: COLORS[k] }}>● <strong>{k}</strong> {COEFF_LABELS[k]}</span>
        ))}
      </div>
      <div style={{ width: "100%", overflowX: "auto" }}>
        <canvas ref={canvasRef} />
      </div>
      <p className="panel-desc">
        Distribution of coefficient values across all wires in each layer.
        Tightly clustered values indicate uniform gate structure.
      </p>
    </div>
  );
}
```

**Step 2: Add to App.jsx**

Import and render below `GateStats`:
```jsx
import CoeffHistograms from "./components/CoeffHistograms";
```

In the Explore mode JSX (after `<GateStats .../>` around line 355):
```jsx
{displayCircuit && <CoeffHistograms circuit={displayCircuit} />}
```

**Step 3: Verify visually**

Run: `npm run dev`
Skip to Explore mode, verify the coefficient histograms appear below Gate Structure Analysis.

**Step 4: Commit**

```bash
git add tools/circuit-explorer/src/components/CoeffHistograms.jsx tools/circuit-explorer/src/App.jsx
git commit -m "feat: add CoeffHistograms component"
```

---

### Task 5: Activation Distribution Ribbon

**Files:**
- Create: `tools/circuit-explorer/src/components/ActivationRibbon.jsx`
- Modify: `tools/circuit-explorer/src/App.jsx`

**Step 1: Create `ActivationRibbon.jsx`**

Canvas-rendered ribbon chart showing per-layer activation statistics: mean, ±σ, ±2σ, and min/max bands.

```jsx
/**
 * ActivationRibbon — Per-layer activation distribution bands.
 * Shows mean ±σ, ±2σ, and min/max as nested colored bands.
 * Canvas-rendered for performance.
 */
import { useEffect, useRef } from "react";

export default function ActivationRibbon({ means, stds, mins, maxs, depth: d, width: n }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!means || !stds || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const W = container.offsetWidth || 600;
    const H = 200;
    const PAD = { top: 10, bottom: 28, left: 44, right: 10 };
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const layers = Math.min(d, means.length);
    const xScale = (l) => PAD.left + (l / Math.max(1, layers - 1)) * plotW;
    const yScale = (v) => PAD.top + (1 - (v + 1) / 2) * plotH; // [-1,1] → [plotH, 0]

    // Compute per-layer aggregate stats
    const agg = [];
    for (let l = 0; l < layers; l++) {
      let sumMean = 0, sumStd = 0, mn = Infinity, mx = -Infinity;
      for (let w = 0; w < n && w < means[l].length; w++) {
        sumMean += means[l][w];
        sumStd += stds[l][w];
        const lo = mins ? mins[l][w] : means[l][w] - stds[l][w];
        const hi = maxs ? maxs[l][w] : means[l][w] + stds[l][w];
        if (lo < mn) mn = lo;
        if (hi > mx) mx = hi;
      }
      const avgMean = sumMean / n;
      const avgStd = sumStd / n;
      agg.push({ mean: avgMean, std: avgStd, min: mn, max: mx });
    }

    // Draw bands: min/max → ±2σ → ±σ → mean line
    const bands = [
      { getY: (a) => [a.min, a.max], color: "rgba(240,82,77,0.08)", label: "min–max" },
      { getY: (a) => [a.mean - 2*a.std, a.mean + 2*a.std], color: "rgba(240,82,77,0.12)", label: "±2σ" },
      { getY: (a) => [a.mean - a.std, a.mean + a.std], color: "rgba(240,82,77,0.22)", label: "±σ" },
    ];

    for (const band of bands) {
      ctx.beginPath();
      for (let l = 0; l < layers; l++) {
        const [lo] = band.getY(agg[l]);
        const x = xScale(l);
        ctx.lineTo(x, yScale(Math.max(-1, lo)));
      }
      for (let l = layers - 1; l >= 0; l--) {
        const [, hi] = band.getY(agg[l]);
        const x = xScale(l);
        ctx.lineTo(x, yScale(Math.min(1, hi)));
      }
      ctx.closePath();
      ctx.fillStyle = band.color;
      ctx.fill();
    }

    // Mean line
    ctx.beginPath();
    ctx.strokeStyle = "#F0524D";
    ctx.lineWidth = 2;
    for (let l = 0; l < layers; l++) {
      const x = xScale(l);
      const y = yScale(agg[l].mean);
      l === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Zero line
    ctx.beginPath();
    ctx.strokeStyle = "rgba(148,163,184,0.4)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.moveTo(PAD.left, yScale(0));
    ctx.lineTo(PAD.left + plotW, yScale(0));
    ctx.stroke();
    ctx.setLineDash([]);

    // Axes
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    const labelStep = Math.max(1, Math.floor(layers / 10));
    for (let l = 0; l < layers; l += labelStep) {
      ctx.fillText(`${l}`, xScale(l), H - 4);
    }
    ctx.fillText("Layer", PAD.left + plotW / 2, H - 14);

    ctx.textAlign = "right";
    ctx.fillText("+1", PAD.left - 4, yScale(1) + 3);
    ctx.fillText("0", PAD.left - 4, yScale(0) + 3);
    ctx.fillText("−1", PAD.left - 4, yScale(-1) + 3);
  }, [means, stds, mins, maxs, d, n]);

  if (!means || !stds) return null;

  return (
    <div className="panel">
      <h2>Activation Distribution</h2>
      <div style={{ width: "100%", overflowX: "auto" }}>
        <canvas ref={canvasRef} />
      </div>
      <div className="formula-legend" style={{ marginTop: 4 }}>
        <span style={{ color: "#F0524D" }}>━ mean</span>
        <span style={{ color: "rgba(240,82,77,0.5)" }}>░ ±σ</span>
        <span style={{ color: "rgba(240,82,77,0.3)" }}>░ ±2σ</span>
        <span style={{ color: "rgba(240,82,77,0.15)" }}>░ min–max</span>
      </div>
    </div>
  );
}
```

**Step 2: Add to App.jsx**

Import and render in the panels-row alongside existing Signal Heatmap:
```jsx
import ActivationRibbon from "./components/ActivationRibbon";
```

In the Explore mode's stats panels area (after the existing `panels-row`):
```jsx
{groundTruthStats?.stds && (
  <ActivationRibbon
    means={groundTruth}
    stds={groundTruthStats.stds}
    mins={groundTruthStats.mins}
    maxs={groundTruthStats.maxs}
    depth={params.depth}
    width={params.width}
  />
)}
```

**Step 3: Verify visually**

Run ground truth estimation, verify ribbon appears with nested colored bands.

**Step 4: Commit**

```bash
git add tools/circuit-explorer/src/components/ActivationRibbon.jsx tools/circuit-explorer/src/App.jsx
git commit -m "feat: add ActivationRibbon component"
```

---

### Task 6: Signal Propagation Std Heatmap

**Files:**
- Create: `tools/circuit-explorer/src/components/StdHeatmap.jsx`
- Modify: `tools/circuit-explorer/src/App.jsx`

**Step 1: Create `StdHeatmap.jsx`**

Canvas heatmap rendering standard deviation per wire per layer. Uses a single-ended color scale: low σ = dark background, high σ = bright coral.

```jsx
/**
 * StdHeatmap — Wire × Layer standard deviation heatmap.
 * Shows where activations are most variable (input-dependent).
 * Low σ = dark (predictable), High σ = bright coral (variable).
 */
import { useEffect, useRef } from "react";

function stdToColor(std, maxStd) {
  const t = Math.min(1, std / Math.max(0.01, maxStd));
  // Dark (#1E293B) → Coral (#F0524D)
  const r = Math.round(30 + (240 - 30) * t);
  const g = Math.round(41 + (82 - 41) * t);
  const b = Math.round(59 + (77 - 59) * t);
  return `rgb(${r},${g},${b})`;
}

export default function StdHeatmap({ stds, width: n, depth: d }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!stds || stds.length === 0 || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const containerW = container.offsetWidth || 500;

    const LABEL_PAD_Y = 28;
    const LABEL_PAD_X = 36;
    const cellW = Math.max(4, Math.floor((containerW - LABEL_PAD_X - 20) / n));
    const cellH = Math.max(4, Math.min(20, Math.floor(280 / d)));
    const chartW = cellW * n;
    const chartH = cellH * d;

    const totalW = chartW + LABEL_PAD_X + 10;
    const totalH = chartH + LABEL_PAD_Y + 10;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = totalW * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = `${totalW}px`;
    canvas.style.height = `${totalH}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, totalW, totalH);

    // Find global max std for normalization
    let maxStd = 0;
    for (let l = 0; l < d && l < stds.length; l++) {
      for (let w = 0; w < n; w++) {
        if (stds[l][w] > maxStd) maxStd = stds[l][w];
      }
    }

    // Draw cells
    for (let l = 0; l < d && l < stds.length; l++) {
      for (let w = 0; w < n; w++) {
        ctx.fillStyle = stdToColor(stds[l][w], maxStd);
        ctx.fillRect(LABEL_PAD_X + w * cellW, l * cellH, cellW - 1, cellH - 1);
      }
    }

    // Axis labels
    ctx.fillStyle = "#64748B";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    const wireLabelStep = n > 16 ? Math.ceil(n / 8) : 1;
    for (let w = 0; w < n; w++) {
      if (w % wireLabelStep === 0) {
        ctx.fillText(`${w}`, LABEL_PAD_X + w * cellW + cellW / 2, chartH + 14);
      }
    }

    const layerLabelStep = d > 16 ? Math.ceil(d / 8) : 1;
    ctx.textAlign = "right";
    for (let l = 0; l < d; l++) {
      if (l % layerLabelStep === 0) {
        ctx.fillText(`${l}`, LABEL_PAD_X - 4, l * cellH + cellH / 2 + 3);
      }
    }

    ctx.fillStyle = "#94A3B8";
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Wire", LABEL_PAD_X + chartW / 2, chartH + 26);
    ctx.save();
    ctx.translate(10, chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Layer", 0, 0);
    ctx.restore();
  }, [stds, n, d]);

  if (!stds || stds.length === 0) return null;

  return (
    <div className="panel">
      <h2>Signal Variability (σ)</h2>
      <div style={{ width: "100%", overflowX: "auto" }}>
        <canvas ref={canvasRef} />
      </div>
      <div className="formula-legend" style={{ marginTop: 4 }}>
        <span style={{ color: "#1E293B" }}>◆ Low σ (predictable)</span>
        <span style={{ color: "#F0524D" }}>◆ High σ (variable)</span>
      </div>
      <p className="panel-desc">
        Wires with high σ are strongly input-dependent.
        Wires with low σ produce nearly constant output regardless of inputs.
      </p>
    </div>
  );
}
```

**Step 2: Add to App.jsx in a panels-row with ActivationRibbon**

```jsx
import StdHeatmap from "./components/StdHeatmap";
```

Render in a `panels-row`:
```jsx
{groundTruthStats?.stds && (
  <div className="panels-row panel-reveal">
    <ActivationRibbon ... />
    <StdHeatmap stds={groundTruthStats.stds} width={params.width} depth={params.depth} />
  </div>
)}
```

**Step 3: Verify, commit**

```bash
git add tools/circuit-explorer/src/components/StdHeatmap.jsx tools/circuit-explorer/src/App.jsx
git commit -m "feat: add StdHeatmap component"
```

---

### Task 7: Mean Propagation Error Heatmap

**Files:**
- Create: `tools/circuit-explorer/src/components/ErrorHeatmap.jsx`
- Modify: `tools/circuit-explorer/src/App.jsx`

**Step 1: Create `ErrorHeatmap.jsx`**

Shows `|groundTruth[l][w] - meanProp[l][w]|` as a canvas heatmap. White = no error, deep red = large error.

```jsx
/**
 * ErrorHeatmap — Mean propagation error per wire per layer.
 * Shows |groundTruth - meanProp| as a heatmap.
 * White = no error, Deep red = large error.
 */
import { useEffect, useMemo, useRef } from "react";

function errorToColor(err, maxErr) {
  const t = Math.min(1, err / Math.max(0.001, maxErr));
  // White (#FFFFFF) → Deep Red (#991B1B)
  const r = Math.round(255 - (255 - 153) * t);
  const g = Math.round(255 - (255 - 27) * t);
  const b = Math.round(255 - (255 - 27) * t);
  return `rgb(${r},${g},${b})`;
}

export default function ErrorHeatmap({ groundTruth, meanPropEstimates, width: n, depth: d }) {
  const canvasRef = useRef(null);

  const { errors, maxErr } = useMemo(() => {
    if (!groundTruth || !meanPropEstimates) return { errors: null, maxErr: 0 };
    const layers = Math.min(d, groundTruth.length, meanPropEstimates.length);
    const errs = [];
    let mx = 0;
    for (let l = 0; l < layers; l++) {
      const layerErr = new Float32Array(n);
      for (let w = 0; w < n; w++) {
        const e = Math.abs((groundTruth[l][w] || 0) - (meanPropEstimates[l][w] || 0));
        layerErr[w] = e;
        if (e > mx) mx = e;
      }
      errs.push(layerErr);
    }
    return { errors: errs, maxErr: mx };
  }, [groundTruth, meanPropEstimates, n, d]);

  useEffect(() => {
    if (!errors || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const containerW = container.offsetWidth || 500;

    const LABEL_PAD_Y = 28;
    const LABEL_PAD_X = 36;
    const cellW = Math.max(4, Math.floor((containerW - LABEL_PAD_X - 20) / n));
    const cellH = Math.max(4, Math.min(20, Math.floor(280 / d)));
    const chartW = cellW * n;
    const chartH = cellH * d;
    const totalW = chartW + LABEL_PAD_X + 10;
    const totalH = chartH + LABEL_PAD_Y + 10;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = totalW * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = `${totalW}px`;
    canvas.style.height = `${totalH}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, totalW, totalH);

    for (let l = 0; l < errors.length; l++) {
      for (let w = 0; w < n; w++) {
        ctx.fillStyle = errorToColor(errors[l][w], maxErr);
        ctx.fillRect(LABEL_PAD_X + w * cellW, l * cellH, cellW - 1, cellH - 1);
      }
    }

    // Axis labels (same pattern as StdHeatmap)
    ctx.fillStyle = "#64748B";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    const wireLabelStep = n > 16 ? Math.ceil(n / 8) : 1;
    for (let w = 0; w < n; w++) {
      if (w % wireLabelStep === 0)
        ctx.fillText(`${w}`, LABEL_PAD_X + w * cellW + cellW / 2, chartH + 14);
    }
    const layerLabelStep = d > 16 ? Math.ceil(d / 8) : 1;
    ctx.textAlign = "right";
    for (let l = 0; l < d && l < errors.length; l++) {
      if (l % layerLabelStep === 0)
        ctx.fillText(`${l}`, LABEL_PAD_X - 4, l * cellH + cellH / 2 + 3);
    }
    ctx.fillStyle = "#94A3B8";
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Wire", LABEL_PAD_X + chartW / 2, chartH + 26);
    ctx.save();
    ctx.translate(10, chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Layer", 0, 0);
    ctx.restore();
  }, [errors, maxErr, n, d]);

  if (!errors) return null;

  return (
    <div className="panel">
      <h2>Mean Propagation Error</h2>
      <div style={{ width: "100%", overflowX: "auto" }}>
        <canvas ref={canvasRef} />
      </div>
      <div className="formula-legend" style={{ marginTop: 4 }}>
        <span style={{ color: "#FFFFFF", background: "#E5E7EB", padding: "0 4px", borderRadius: 2 }}>◆ No error</span>
        <span style={{ color: "#991B1B" }}>◆ High |GT − MP|</span>
        <span style={{ color: "#9CA3AF", fontSize: 10 }}> max = {maxErr.toFixed(4)}</span>
      </div>
      <p className="panel-desc">
        Where does mean propagation break? Gates with <strong>product terms</strong> (p·x·y)
        violate the independence assumption, especially when upstream wires share inputs.
      </p>
    </div>
  );
}
```

**Step 2: Add to App.jsx**

```jsx
import ErrorHeatmap from "./components/ErrorHeatmap";
```

Render when both ground truth and mean prop exist (after EstimatorComparison):
```jsx
{groundTruth && meanPropEst && (
  <ErrorHeatmap
    groundTruth={groundTruth}
    meanPropEstimates={meanPropEst}
    width={params.width}
    depth={params.depth}
  />
)}
```

**Step 3: Verify, commit**

```bash
git add tools/circuit-explorer/src/components/ErrorHeatmap.jsx tools/circuit-explorer/src/App.jsx
git commit -m "feat: add ErrorHeatmap component"
```

---

### Task 8: Animated Signal Flow

**Files:**
- Create: `tools/circuit-explorer/src/components/SignalAnimation.jsx`
- Modify: `tools/circuit-explorer/src/App.jsx`
- Modify: `tools/circuit-explorer/src/App.css`

**Step 1: Create `SignalAnimation.jsx`**

A hook + button component that manages animation state for signal flow.

```jsx
/**
 * SignalAnimation — Animated signal flow controller.
 * Graph mode: animates JointJS gate fills layer by layer.
 * Heatmap mode: sweeping wavefront across the heatmap.
 *
 * Provides: { isAnimating, animLayer, startAnimation, trialValues }
 */
import { useCallback, useEffect, useRef, useState } from "react";

export function useSignalAnimation(circuit, depth) {
  const [isAnimating, setIsAnimating] = useState(false);
  const [animLayer, setAnimLayer] = useState(-1);
  const [trialValues, setTrialValues] = useState(null);
  const timerRef = useRef(null);
  const seedRef = useRef(1);

  const startAnimation = useCallback(() => {
    if (!circuit || isAnimating) return;

    // Import dynamically to avoid circular deps
    import("../circuit").then(({ runSingleTrial }) => {
      const seed = seedRef.current++;
      const values = runSingleTrial(circuit, seed);
      setTrialValues(values);
      setAnimLayer(-1);
      setIsAnimating(true);

      let layer = 0;
      const tick = () => {
        setAnimLayer(layer);
        layer++;
        if (layer < depth) {
          timerRef.current = setTimeout(tick, Math.max(80, 400 / depth));
        } else {
          timerRef.current = setTimeout(() => {
            setIsAnimating(false);
          }, 600);
        }
      };
      timerRef.current = setTimeout(tick, 200);
    });
  }, [circuit, depth, isAnimating]);

  // Cleanup on unmount
  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  return { isAnimating, animLayer, trialValues, startAnimation };
}

export default function SignalAnimationButton({ onAnimate, isAnimating }) {
  return (
    <button
      className="animate-btn"
      onClick={onAnimate}
      disabled={isAnimating}
      title="Animate a single random input flowing through the circuit"
    >
      {isAnimating ? (
        <span className="animate-btn-pulse">◉ Flowing…</span>
      ) : (
        <>
          <svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
          </svg>
          Animate Flow
        </>
      )}
    </button>
  );
}
```

**Step 2: Add heatmap wavefront overlay to `CircuitHeatmap.jsx`**

Add an optional `animLayer` prop. When set, draw a bright vertical line at the animated layer position:

In `drawCrosshair` callback, add before the existing crosshair code:
```js
// Animation wavefront
if (animLayer >= 0 && animLayer < d) {
  ctx.fillStyle = "rgba(240, 82, 77, 0.15)";
  ctx.fillRect(0, 0, (animLayer + 1) * dims.cellW, dims.height);
  ctx.strokeStyle = "rgba(240, 82, 77, 0.8)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo((animLayer + 1) * dims.cellW, 0);
  ctx.lineTo((animLayer + 1) * dims.cellW, dims.height);
  ctx.stroke();
}
```

**Step 3: Wire up in App.jsx**

```jsx
import SignalAnimationButton, { useSignalAnimation } from "./components/SignalAnimation";
```

In App component body:
```jsx
const { isAnimating, animLayer, trialValues, startAnimation } = useSignalAnimation(displayCircuit, effectiveParams.depth);
```

Add the Animate button near the circuit visualization header, and pass `animLayer` to `CircuitHeatmap` / `CircuitGraphJoint`.

**Step 4: Add CSS for animate button**

```css
.animate-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  font-size: 11px;
  font-family: var(--font-mono);
  color: #F0524D;
  background: rgba(240, 82, 77, 0.08);
  border: 1px solid rgba(240, 82, 77, 0.2);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s;
}
.animate-btn:hover:not(:disabled) {
  background: rgba(240, 82, 77, 0.15);
  border-color: rgba(240, 82, 77, 0.4);
}
.animate-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
.animate-btn-pulse {
  animation: pulse-flow 0.8s ease-in-out infinite alternate;
}
@keyframes pulse-flow {
  from { opacity: 0.5; }
  to { opacity: 1; }
}
```

**Step 5: Verify — click Animate Flow, watch wavefront sweep**

**Step 6: Commit**

```bash
git add tools/circuit-explorer/src/components/SignalAnimation.jsx \
  tools/circuit-explorer/src/components/CircuitHeatmap.jsx \
  tools/circuit-explorer/src/App.jsx tools/circuit-explorer/src/App.css
git commit -m "feat: add animated signal flow"
```

---

### Task 9: Final Integration & Polish

**Files:**
- Modify: `tools/circuit-explorer/src/App.jsx` (final layout ordering)
- Modify: `tools/circuit-explorer/src/App.css` (any missing styles)

**Step 1: Ensure panel ordering in Explore mode**

Final layout order in Explore mode JSX:
1. Circuit Graph/Heatmap (existing) + Animate button
2. Gate Structure Analysis (existing `GateStats`)
3. Coefficient Histograms (new `CoeffHistograms`)
4. Row: Signal Heatmap (existing) + Wire Stats (existing)
5. Row: Activation Ribbon + Std Heatmap (new, require stats)
6. Mean Prop Error Heatmap (new, requires GT + MP)
7. Estimation Error MSE (existing `EstimatorComparison`)

**Step 2: Verify full dashboard flow**

1. Open app, skip tour to Explore mode
2. Set width=8, depth=6
3. Click Ground Truth → verify CoeffHistograms (always visible), then Activation Ribbon + Std Heatmap appear
4. Click Mean Propagation → verify Error Heatmap appears
5. Click Animate Flow → verify wavefront animation
6. Increase to width=64, depth=32 (heatmap mode) → verify all panels scale correctly
7. Increase to width=256, depth=128 → verify performance is acceptable

**Step 3: Commit final integration**

```bash
git add -A tools/circuit-explorer/src/
git commit -m "feat: finalize advanced dashboard visualizations layout"
```
