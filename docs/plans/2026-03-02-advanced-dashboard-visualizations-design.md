# Advanced Dashboard Visualizations

Enrich the Circuit Explorer dashboard with 5 new visualization components inspired by TensorBoard, tfjs-vis, and neural network activation analysis. The goal: make circuit internals *feel* as explorable as a neural network's weights and activations.

## Selected Visualizations

### 1. Layer-by-Layer Coefficient Histograms (TensorBoard-style)

**What:** A row of small histograms — one per layer — showing the *distribution* of coefficient values (c, a, b, p) across all wires in that layer. Unlike the current `GateStats` which only counts "dominant coefficient", this shows the actual *spread* and *clustering* of values.

**Why:** Reveals structural patterns: are gates tightly clustered at ±0.5 (complex AND gates), or spread uniformly? Directly parallels TensorBoard weight histograms.

**Rendering:** Custom Canvas sparklines (4 mini-histograms per layer, color-coded by coefficient type). tfvis.render.histogram requires a tf.Tensor and is heavy — inline canvas sparklines will be faster and denser.

**Data needed:** Circuit gate coefficients (already available).

---

### 2. Activation Distribution Ribbon

**What:** A stacked ridge of distribution summaries per layer: for each layer, show the spread of wire output values across all trials. Renders as a "ribbon" showing μ, μ±σ, μ±2σ, and min/max bands — like a candlestick chart for activations.

**Why:** Shows whether activations are bimodal (±1 dominant at inputs), collapsing to zero, or diverging. This is the "health check" for information flow.

**Rendering:** Custom Canvas — stacked bands per layer with gradient fills. More performant than tfvis for dense layer counts.

**Data needed:** Per-wire per-layer std, min, max (new — computed alongside existing mean).

---

### 3. Signal Propagation Std Heatmap

**What:** A 2D canvas heatmap of standard deviations per wire per layer (analogous to the existing wire-means heatmap but for variability). Low std → wire is predictable. High std → wire is highly input-dependent.

**Why:** Reveals "information flow": which wires carry the most signal vs. which are effectively constant. Directly complementary to the means heatmap.

**Rendering:** Canvas heatmap (reuse `CircuitHeatmap`/`SignalHeatmap` patterns). Diverging color scale: low σ = cool/muted, high σ = warm/intense.

**Data needed:** Per-wire per-layer std (new — from enriched pipeline).

---

### 4. Mean Propagation Error Heatmap

**What:** A 2D heatmap showing `|groundTruth[l][w] - meanProp[l][w]|` per wire per layer. Highlights exactly where mean propagation's independence assumption breaks down.

**Why:** This is the most challenge-relevant visualization. Participants can see *which gates* cause the most estimation error, and correlate with the coefficient fingerprint (AND gates with product terms disrupt independence).

**Rendering:** Canvas heatmap, same style as Std Heatmap. Single-ended color scale (0 = white, high error = deep red).

**Data needed:** Requires both ground truth and mean propagation results. Both already available via `EstimatorRunner`.

---

### 5. Animated Signal Flow

**What:** Animate the propagation of a single random ±1 input through the circuit, one layer at a time. 

- **Graph mode:** Color each gate-output wire as its layer is "reached", with a brief pulse animation. Wires transition from gray → computed color over ~200ms.
- **Heatmap mode:** A vertical "wavefront" line sweeps left-to-right across the heatmap, progressively revealing wire values column-by-column.

**Why:** Makes the circuit feel alive. Users intuitively grasp how values propagate through layers.

**Rendering:** 
- Graph mode: animate JointJS element fill via `requestAnimationFrame` loop, layer by layer.
- Heatmap mode: overlay canvas animation with a sweeping reveal mask.

**Data needed:** `runBatched` with a single input (1 trial). The data is tiny — one `Float32Array[n]` per layer.

---

## Data Pipeline Changes

### Enriched `empiricalMean` → `empiricalStats`

Modify `empiricalMean` in `circuit.js` to compute and return richer per-layer stats alongside means:

```js
// Returns: { means: Float32Array[], stds: Float32Array[], mins: Float32Array[], maxs: Float32Array[] }
export function empiricalStats(circuit, trials, seed = 99)
```

**Changes:**
- `circuit.js`: New `empiricalStats()` function (keeps `empiricalMean` unchanged for backward compat)
- `circuit-tf.js`: New `empiricalStatsTF()` — compute mean, variance, min, max using tf.moments / tf.min / tf.max per layer
- `circuit.worker.js`: Add `'empiricalStats'` message handler
- `EstimatorRunner.jsx`: Call `empiricalStats` instead of `empiricalMean` for ground truth, pass richer data upstream
- `App.jsx`: Thread `stats` (std/min/max) down to new visualization components

### Single-Trial Run for Animation

Add a `runSingleTrial` convenience in `circuit.js`:
```js
// Returns: Float32Array[] — one per layer, each of length n
export function runSingleTrial(circuit, seed = 42)
```

## Dashboard Layout

New visualizations appear in the Explore mode panel area below the existing panels:

```
┌─ Circuit Graph/Heatmap ─────────────────────┐
│  [existing]                  [▶ Animate btn] │
└──────────────────────────────────────────────┘

┌─ Gate Structure Analysis ───────────────────┐
│  [existing bar charts]                       │
└──────────────────────────────────────────────┘

┌─ Coefficient Histograms ─ NEW ──────────────┐
│  [L0] [L1] [L2] ... [Ld-1] small sparklines │
└──────────────────────────────────────────────┘

┌─── panels-row ──────────────────────────────┐
│ ┌─ Signal Heatmap ─┐ ┌─ Wire Stats ────────┐│
│ │  [existing]       │ │  [existing]         ││
│ └───────────────────┘ └─────────────────────┘│
└──────────────────────────────────────────────┘

┌─── panels-row ──────────────────────────────┐
│ ┌─ Activation Ribbon ─ NEW ─┐ ┌─ Std Hmap ─┐│
│ │  [ribbon chart]            │ │  [heatmap] ││
│ └────────────────────────────┘ └────────────┘│
└──────────────────────────────────────────────┘

┌─ Mean Prop Error Heatmap ─ NEW ─────────────┐
│  [error heatmap — only when both GT + MP]    │
└──────────────────────────────────────────────┘

┌─ Estimation Error (MSE per Layer) ──────────┐
│  [existing comparison chart]                 │
└──────────────────────────────────────────────┘
```

## New Files

| File | Type | Description |
|------|------|-------------|
| `CoeffHistograms.jsx` | Component | Layer-by-layer coefficient distribution sparklines |
| `ActivationRibbon.jsx` | Component | Per-layer activation distribution ribbon chart |
| `StdHeatmap.jsx` | Component | Wire × layer standard deviation heatmap |
| `ErrorHeatmap.jsx` | Component | Mean propagation error heatmap |
| `SignalAnimation.jsx` | Component | Animated signal flow controller + overlay |

## Modified Files

| File | Change |
|------|--------|
| `circuit.js` | Add `empiricalStats()`, `runSingleTrial()` |
| `circuit-tf.js` | Add `empiricalStatsTF()` |
| `circuit.worker.js` | Add `'empiricalStats'` handler |
| `EstimatorRunner.jsx` | Use `empiricalStats` for ground truth |
| `App.jsx` | Thread stats data, add animation state, render new panels |
| `App.css` | Styles for new components |
