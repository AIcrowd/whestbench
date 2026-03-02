# Gate Type Visualization — Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Add gate type awareness (metadata, symbols in circuit view, error-by-gate-type chart) so users can see individual boolean gate identities and analyze how estimation error varies by gate type.

**Architecture:** Expand the existing `classifyGate()` in `gateShapes.js` to return a full boolean classification with symbol and label. Render the symbol inside existing JointJS rectangles in small-circuit mode. Add a new `ErrorByGateType` canvas chart component following the `EstimatorComparison` pattern.

**Tech Stack:** React, JointJS (existing), Canvas 2D (chart), existing `circuit.js` / `gateShapes.js`

---

### Task 1: Expand Gate Classifier in `gateShapes.js`

**Files:**
- Modify: `tools/circuit-explorer/src/components/gateShapes.js`
- Modify: `tools/circuit-explorer/src/circuit.js` (remove `describeGate`)

**Step 1: Rewrite `classifyGate()` to return full gate identity**

Replace the current 4-type classifier with full boolean classification. The function should return `{ type, label, symbol }`.

```js
/**
 * GATE_TYPES — canonical gate definitions.
 * Each gate gets a short type key, a unicode symbol, and a color.
 */
export const GATE_TYPES = {
  BUF:   { symbol: "▷",  color: "#6366F1" },  // indigo
  NOT:   { symbol: "▷○", color: "#8B5CF6" },  // violet
  CONST: { symbol: "■",  color: "#9CA3AF" },  // gray
  XOR:   { symbol: "⊕",  color: "#F59E0B" },  // amber
  XNOR:  { symbol: "⊙",  color: "#D97706" },  // dark amber
  AND:   { symbol: "∧",  color: "#10B981" },  // emerald
  NAND:  { symbol: "∧̄",  color: "#059669" },  // dark emerald
  OR:    { symbol: "∨",  color: "#3B82F6" },  // blue
  NOR:   { symbol: "∨̄",  color: "#2563EB" },  // dark blue
};

/**
 * Classify a gate by its coefficient pattern into one of the 10 boolean
 * function families. Returns { type, label, symbol }.
 *
 * type:   key from GATE_TYPES (e.g. "AND", "XOR")
 * label:  human-readable with input signs (e.g. "AND(-x, y)")
 * symbol: unicode motif (e.g. "∧")
 */
export function classifyGate(layer, wireIndex) {
  const c = layer.const[wireIndex];
  const a = layer.firstCoeff[wireIndex];
  const b = layer.secondCoeff[wireIndex];
  const p = layer.productCoeff[wireIndex];

  const hasC = Math.abs(c) > 1e-6;
  const hasA = Math.abs(a) > 1e-6;
  const hasB = Math.abs(b) > 1e-6;
  const hasP = Math.abs(p) > 1e-6;

  // --- Simple gates: exactly one nonzero coefficient ---
  if (!hasC && (hasA || hasB) && !hasP && !(hasA && hasB)) {
    if (hasA) return { type: a > 0 ? "BUF" : "NOT", label: a > 0 ? "x" : "−x", symbol: GATE_TYPES[a > 0 ? "BUF" : "NOT"].symbol };
    return { type: b > 0 ? "BUF" : "NOT", label: b > 0 ? "y" : "−y", symbol: GATE_TYPES[b > 0 ? "BUF" : "NOT"].symbol };
  }
  if (hasC && !hasA && !hasB && !hasP) {
    return { type: "CONST", label: c > 0 ? "+1" : "−1", symbol: GATE_TYPES.CONST.symbol };
  }
  if (!hasC && !hasA && !hasB && hasP) {
    return { type: p > 0 ? "XNOR" : "XOR", label: p > 0 ? "XNOR(x,y)" : "XOR(x,y)", symbol: GATE_TYPES[p > 0 ? "XNOR" : "XOR"].symbol };
  }

  // --- Complex gates: AND family (all four coeffs involved) ---
  // The gate is ±AND(±x, ±y).
  // Determine input signs and outer sign from coefficient ratios.
  const xSign = (a / p) > 0 ? "" : "−";
  const ySign = (b > 0) === (p > 0) ? "" : "−";
  const outerPositive = (c < 0) === (p > 0);

  // Derive the boolean function from outerSign and input signs
  // AND(x,y) coeffs: c=-0.5, a=0.5, b=0.5, p=0.5
  // -AND(x,y) = NOR(-x,-y) — De Morgan's
  // AND(-x,-y) = NOR(x,y) — same
  const xNeg = xSign === "−";
  const yNeg = ySign === "−";

  let type, label;
  if (outerPositive) {
    // +AND(±x, ±y)
    if (!xNeg && !yNeg) { type = "AND";  label = "AND(x, y)"; }
    else if (xNeg && yNeg) { type = "NOR";  label = "NOR(x, y)"; }
    else { type = "AND"; label = `AND(${xSign}x, ${ySign}y)`; }
  } else {
    // -AND(±x, ±y) = NAND(±x, ±y)
    if (!xNeg && !yNeg) { type = "NAND"; label = "NAND(x, y)"; }
    else if (xNeg && yNeg) { type = "OR";   label = "OR(x, y)"; }
    else { type = "NAND"; label = `NAND(${xSign}x, ${ySign}y)`; }
  }

  return { type, label, symbol: GATE_TYPES[type].symbol };
}
```

**Step 2: Remove `describeGate()` from `circuit.js`**

Delete lines 232-262 of `circuit.js` (the `describeGate` function and its JSDoc).

**Step 3: Update `CircuitGraph.jsx` import**

Change the import on line 2 of `CircuitGraph.jsx` from:
```js
import { describeGate } from "../circuit";
```
to:
```js
import { classifyGate } from "./gateShapes";
```

And update the usage on line 135 from `describeGate(layer, i)` to `classifyGate(layer, i).label`.

**Step 4: Update `GateDetailOverlay.jsx`**

The existing `classifyGate` call on line 96 returns `{ type }` — no changes needed to the call, but the returned object now has richer data. If `GateDetailOverlay` uses `.type` for colors via `gateColor()`, update `gateColor()` to handle the new type keys:

```js
export function gateColor(type) {
  return GATE_TYPES[type] || GATE_TYPES.AND;
}
```

(Returns `{ symbol, color }` instead of `{ stroke, text }`.)

If `GateDetailOverlay` destructures `{ stroke, text }` from `gateColor()`, update those references to use `{ color }` instead.

**Step 5: Commit**

```bash
git add tools/circuit-explorer/src/components/gateShapes.js \
       tools/circuit-explorer/src/circuit.js \
       tools/circuit-explorer/src/components/CircuitGraph.jsx \
       tools/circuit-explorer/src/components/GateDetailOverlay.jsx
git commit -m "feat: expand gate classifier to full boolean type taxonomy"
```

---

### Task 2: Render Gate Symbols in CircuitGraphJoint

**Files:**
- Modify: `tools/circuit-explorer/src/components/CircuitGraphJoint.jsx`

**Step 1: Import `classifyGate` and `GATE_TYPES`**

Add to the imports at top:
```js
import { classifyGate, GATE_TYPES, GATE_H, GATE_OPACITY, GATE_W, INPUT_DOT_R, meanToColor, WIRE_PORT_R } from "./gateShapes";
```

**Step 2: Add gate symbol text to each rectangle**

In the gate creation loop (around line 289), after creating the `shapes.standard.Rectangle`, add the gate symbol as a label:

```js
const gateInfo = classifyGate(circuit.gates[l], w);

const node = new shapes.standard.Rectangle({
  position: { x, y },
  size: { width: GATE_W, height: GATE_H },
  attrs: {
    body: {
      fill: GATE_FILL_DEFAULT,
      stroke: GATE_STROKE,
      strokeWidth: 1.5,
      rx: 3,
      ry: 3,
      opacity: GATE_OPACITY,
    },
    label: {
      text: gateInfo.symbol,
      fontSize: 10,
      fontFamily: "system-ui, sans-serif",
      fill: "#475569",
      textAnchor: "middle",
      textVerticalAnchor: "middle",
    },
  },
  // ... ports same as before
});
```

Also store `gateInfo` in `gateData`:
```js
node.set("gateData", {
  ...existingFields,
  gateType: gateInfo.type,
  gateLabel: gateInfo.label,
  gateSymbol: gateInfo.symbol,
});
```

**Step 3: Add gate type row to tooltip**

In the tooltip section (around line 677), add a gate type row right after the header:

```jsx
<div className="canvas-tip-rows">
  <div className="canvas-tip-row">
    <span className="canvas-tip-label">Gate type</span>
    <span className="canvas-tip-value" style={{ fontWeight: 600 }}>
      {tooltip.gateLabel}
    </span>
  </div>
</div>
<div className="canvas-tip-divider" />
```

**Step 4: Add gate type legend below circuit header**

In the legend area (around line 621), add a gate type legend row:

```jsx
<div className="gate-legend" style={{ fontSize: 10, color: "#64748B", flexWrap: "wrap", gap: "4px 12px" }}>
  {Object.entries(GATE_TYPES).map(([key, { symbol, color }]) => (
    <span key={key} style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
      <span style={{ color, fontWeight: 700, fontSize: 12 }}>{symbol}</span>
      <span>{key}</span>
    </span>
  ))}
</div>
```

**Step 5: Commit**

```bash
git add tools/circuit-explorer/src/components/CircuitGraphJoint.jsx
git commit -m "feat: render gate type symbols inside circuit rectangles with legend"
```

---

### Task 3: New `ErrorByGateType` Chart Component

**Files:**
- Create: `tools/circuit-explorer/src/components/ErrorByGateType.jsx`
- Modify: `tools/circuit-explorer/src/App.jsx`

**Step 1: Create `ErrorByGateType.jsx`**

Canvas-rendered grouped bar chart following the `EstimatorComparison` pattern:

```jsx
/**
 * ErrorByGateType — grouped bar chart showing mean |error| by gate type.
 * Canvas-rendered with CanvasTooltip on hover.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CanvasTooltip from "./CanvasTooltip";
import InfoTip from "./InfoTip";
import { classifyGate, GATE_TYPES } from "./gateShapes";

const SERIES_COLORS = {
  sampling: "#F0524D",
  meanProp: "#94A3B8",
  covProp:  "#2DD4BF",
};

export default function ErrorByGateType({
  circuit,
  groundTruth,
  samplingEstimates,
  meanPropEstimates,
  covPropEstimates,
  activeLayer,
}) {
  // Implementation:
  // 1. For the active layer (or all layers if none), classify each gate
  // 2. Group by gate type
  // 3. For each type, compute mean |error| = mean(|estimate[l][w] - groundTruth[l][w]|) per estimator
  // 4. Draw grouped bars: one group per gate type, one bar per estimator
  // 5. X-axis: gate type labels with symbols
  // 6. Y-axis: mean |error|
  // 7. Hover tooltip with exact values
  //
  // Follow EstimatorComparison.jsx canvas pattern:
  // - useRef for canvas
  // - useMemo for data computation
  // - useEffect for canvas drawing
  // - CanvasTooltip for hover

  // ... (full implementation ~150-200 lines following EstimatorComparison pattern)
}
```

The component:
- Props match `EstimatorComparison` + `circuit` object
- Uses `classifyGate()` to classify each gate at the active layer
- Groups gates by type, computes mean |error| per group per estimator
- Draws grouped bars with `GATE_TYPES[type].color` for x-axis labels
- Includes `CanvasTooltip` on hover
- Includes `InfoTip` with explanation
- Uses existing `SERIES_COLORS` for consistency with other charts

**Step 2: Integrate into `App.jsx`**

Add import:
```js
import ErrorByGateType from "./components/ErrorByGateType";
```

Add component after the `EstimatorComparison` block (around line 403), inside the same conditional:

```jsx
{groundTruth && (samplingEst || meanPropEst) && displayCircuit && (
  <div className="panel-reveal">
    <ErrorByGateType
      circuit={displayCircuit}
      groundTruth={groundTruth}
      samplingEstimates={samplingEst}
      meanPropEstimates={meanPropEst}
      covPropEstimates={covPropEst}
      activeLayer={activeLayer}
    />
  </div>
)}
```

**Step 3: Commit**

```bash
git add tools/circuit-explorer/src/components/ErrorByGateType.jsx \
       tools/circuit-explorer/src/App.jsx
git commit -m "feat: add ErrorByGateType grouped bar chart"
```

---

### Task 4: Visual Polish & Verification

**Files:**
- Modify: `tools/circuit-explorer/src/App.css` (if needed for legend/chart styles)

**Step 1: Adjust gate symbol sizing and layout**

After integration, verify that:
- Gate symbols fit inside the 20×32px rectangles without overlap
- The legend row doesn't overflow on narrow screens
- The ErrorByGateType chart renders at reasonable dimensions

Adjust `GATE_W` in `gateShapes.js` if symbols get cut off (may need to widen from 20 to ~28px).

**Step 2: Manual browser verification**

1. Open `http://localhost:5173/`
2. Skip the tour to enter explore mode
3. **Small circuit (default 8×6)**: Verify each gate rectangle shows a symbol inside it. Click a gate and verify the tooltip shows "Gate type: AND(x, y)" (or similar). Verify the legend row shows all gate types with their symbols.
4. **Estimators**: Run Ground Truth, then Sampling + Mean Propagation. Verify the ErrorByGateType chart appears with grouped bars. Hover a bar and verify the tooltip shows the gate type and mean |error|.
5. **Large circuit (set width=100, depth=50)**: Verify it switches to heatmap mode and the gate symbols are NOT shown (heatmap mode uses `CircuitHeatmap`, not `CircuitGraphJoint`).
6. **Layer selection**: Click a layer in the EstimatorComparison chart, verify the ErrorByGateType chart updates to show error for just that layer.

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: polish gate type visualization and verify"
```
