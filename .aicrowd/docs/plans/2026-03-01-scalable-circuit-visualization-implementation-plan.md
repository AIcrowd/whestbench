# Scalable Circuit Visualization Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Replace the current hand-rolled SVG circuit graph with JointJS for small circuits and a heatmap dashboard with detail-overlay hover for large circuits.

**Architecture:** Dual-mode rendering: JointJS interactive graph (with circuit-style gates, zoom/pan) when n×d ≤ 4096, heatmap grid with hover detail overlay (input band → gate element → output band) when n×d > 4096. Analytics panels (Recharts) work at any size. In-UI documentation guides new participants.

**Tech Stack:** `@joint/core` (circuit graph), `recharts` (charts, already installed), React, Vite

---

### Task 1: Install JointJS and Verify Setup

**Files:**
- Modify: `tools/circuit-explorer/package.json`

**Step 1: Install @joint/core**

```bash
cd tools/circuit-explorer && npm install @joint/core
```

**Step 2: Verify it imports correctly**

Create a quick smoke test — add to the top of `App.jsx` temporarily:
```javascript
import * as joint from '@joint/core';
console.log('JointJS version:', joint.version);
```

**Step 3: Run dev server and check console**

```bash
cd tools/circuit-explorer && npm run dev
```
Expected: Console shows JointJS version, no import errors.

**Step 4: Remove smoke test and commit**

```bash
cd tools/circuit-explorer
git add package.json package-lock.json
git commit -m "feat: add @joint/core dependency for interactive circuit graph"
```

---

### Task 2: Build JointJS Circuit Graph Component

**Files:**
- Create: `tools/circuit-explorer/src/components/CircuitGraphJoint.jsx`
- Create: `tools/circuit-explorer/src/components/gateShapes.js`

**Step 1: Create custom gate shape definitions**

Create `tools/circuit-explorer/src/components/gateShapes.js`:

```javascript
// Define custom JointJS shapes for each gate type:
// - BufferGate (triangle): for linear passthrough (a≠0 or b≠0 only)
// - ConstGate (small square): for constant (c≠0 only)
// - ProductGate (circle with ×): for pure product (p≠0 only)
// - AndGate (D-shape): for complex gates (all 4 coeffs set)
//
// Each shape should:
// - Accept a `gateData` attribute with coefficients (c, a, b, p)
// - Accept a `wireMean` attribute for color fill
// - Have 2 input ports (left side) and 1 output port (right side)
// - Show a label with the operation type
//
// Use joint.dia.Element.define() for each shape.
// Color scale: blue (mean=-1) → white (mean=0) → red (mean=1)
```

This file should export a `classifyGate(layer, wireIndex)` function that returns the gate type and label, plus the shape definitions registered with JointJS.

**Step 2: Create the JointJS circuit graph component**

Create `tools/circuit-explorer/src/components/CircuitGraphJoint.jsx`:

```javascript
// This component renders the circuit as an interactive JointJS diagram.
//
// Props:
// - circuit: { n, d, gates[] } — the circuit data
// - means: number[][] — wire means from estimator (optional)
//
// Behavior:
// - Creates a joint.dia.Graph and joint.dia.Paper
// - For each layer l, wire w: creates a gate element using the appropriate shape
// - For each gate: adds links from input wires (layer l-1) to the gate
// - Layout: layers as vertical columns, left to right
// - Spacing: computed from n and d to fill available area
// - Zoom/pan: paper.scale() on mouse wheel, translate on drag
// - Click gate: show tooltip with coefficients and wire mean
// - When `means` changes: update gate fill colors
//
// The paper div should have ref and be sized to fill its parent container.
```

**Step 3: Run dev server, verify graph renders for a small circuit (n=8, d=4)**

Expected: 32 gate nodes arranged in 4 columns, connected by wires, zoomable.

**Step 4: Commit**

```bash
git add tools/circuit-explorer/src/components/gateShapes.js tools/circuit-explorer/src/components/CircuitGraphJoint.jsx
git commit -m "feat: add JointJS circuit graph with custom gate shapes"
```

---

### Task 3: Build Heatmap with Detail-Overlay Hover Component

**Files:**
- Create: `tools/circuit-explorer/src/components/CircuitHeatmap.jsx`
- Create: `tools/circuit-explorer/src/components/GateDetailOverlay.jsx`

**Step 1: Build the base heatmap component**

Create `tools/circuit-explorer/src/components/CircuitHeatmap.jsx`:

```javascript
// Renders the circuit as a wires × layers heatmap grid.
//
// Props:
// - circuit: { n, d, gates[] }
// - means: number[][] — wire means (optional, shows gray if absent)
// - onCellHover: (wire, layer) => void — callback for hover
//
// Rendering strategy:
// - Use a single <canvas> element for performance at 1024×256
// - Each cell colored by wire mean (blue → white → red diverging)
// - X-axis = layers (0..d-1), Y-axis = wires (0..n-1)
// - On mouse move: compute which cell is hovered from mouse coordinates
// - On hover: trigger onCellHover with wire/layer indices
//
// Canvas is preferred over SVG rects for 262k+ cells.
// Use requestAnimationFrame for smooth hover highlighting.
```

**Step 2: Build the gate detail overlay**

Create `tools/circuit-explorer/src/components/GateDetailOverlay.jsx`:

```javascript
// Appears on hover over the heatmap, showing:
// - Left band: 1D vertical heatmap of layer l-1 wire means
//   - Input wires (first[w], second[w]) accented with ▶ markers
// - Center: gate element showing operation and coefficients
// - Right band: 1D vertical heatmap of layer l+1 wire means
//   - Output consumers accented with ◀ markers
//
// Props:
// - circuit: the circuit data
// - means: wire means
// - hoveredWire: number
// - hoveredLayer: number
// - position: { x, y } — where to render the overlay
//
// The overlay should be absolutely positioned near the hovered cell.
// Bands are rendered as thin SVG rects (one per wire).
// Gate element shows: operation label, coefficients, mean value.
```

**Step 3: Verify heatmap renders for a large circuit (n=256, d=64)**

Expected: Dense colored grid, hovering a cell shows the detail overlay.

**Step 4: Commit**

```bash
git add tools/circuit-explorer/src/components/CircuitHeatmap.jsx tools/circuit-explorer/src/components/GateDetailOverlay.jsx
git commit -m "feat: add scalable heatmap with detail-overlay hover"
```

---

### Task 4: Integrate Adaptive Mode Switching in App.jsx

**Files:**
- Modify: `tools/circuit-explorer/src/App.jsx`
- Modify: `tools/circuit-explorer/src/App.css`

**Step 1: Add mode computation and conditional rendering**

In `App.jsx`, compute the rendering mode:
```javascript
const totalGates = params.width * params.depth;
const useJointJS = totalGates <= 4096;
```

Render `CircuitGraphJoint` when `useJointJS` is true, `CircuitHeatmap` otherwise.

Remove the old `CircuitGraph.jsx` import and usage.

**Step 2: Add a mode indicator badge**

Show a small badge in the header: "Graph Mode (n×d = 2048)" or "Heatmap Mode (n×d = 65536)" so users know which mode they're in and why.

**Step 3: Update CSS for both modes**

Ensure both components fill the same layout area. The JointJS paper needs a fixed-height container. The heatmap canvas needs to be responsive.

**Step 4: Verify mode switching**

- Set width=64, depth=32 → should show JointJS graph
- Set width=128, depth=64 → should switch to heatmap
- Verify no flicker or layout jump on transition

**Step 5: Commit**

```bash
git add tools/circuit-explorer/src/App.jsx tools/circuit-explorer/src/App.css
git commit -m "feat: adaptive circuit view — JointJS for small, heatmap for large"
```

---

### Task 5: Build Error Heatmap Panel

**Files:**
- Create: `tools/circuit-explorer/src/components/ErrorHeatmap.jsx`

**Step 1: Create the error heatmap component**

```javascript
// Shows |ground_truth[l][w] - estimate[l][w]| as a heatmap.
//
// Props:
// - groundTruth: number[][] — wire means from ground truth estimator
// - estimate: number[][] — wire means from sampling or mean prop
// - width: number, depth: number
//
// Color scale: white (0 error) → orange → red (high error)
// Uses same canvas rendering strategy as CircuitHeatmap.
// Only renders when both groundTruth and at least one estimate exist.
// Shows per-layer aggregate error bar below.
```

**Step 2: Integrate into App.jsx**

Add below the main circuit view, only visible when ground truth + another estimator have been run.

**Step 3: Verify with a circuit where mean propagation has visible error**

Expected: Error heatmap shows which layers/wires have the worst estimation error.

**Step 4: Commit**

```bash
git add tools/circuit-explorer/src/components/ErrorHeatmap.jsx tools/circuit-explorer/src/App.jsx
git commit -m "feat: add error heatmap showing |GT - estimate| per wire"
```

---

### Task 6: Add In-UI Documentation

**Files:**
- Create: `tools/circuit-explorer/src/components/PanelHelp.jsx`
- Create: `tools/circuit-explorer/src/components/Onboarding.jsx`
- Modify: `tools/circuit-explorer/src/App.jsx`
- Modify: `tools/circuit-explorer/src/App.css`

**Step 1: Create PanelHelp component**

```javascript
// A small ℹ️ icon that expands a help text when clicked.
// Props:
// - title: string
// - children: help text content
// Renders as an icon button that toggles a collapsible explanation.
```

**Step 2: Create Onboarding component**

```javascript
// Shown when no estimator has been run yet.
// Brief guide:
// 1. "This circuit has N wires and D layers of random gates"
// 2. "Your goal: estimate the mean output of each wire"
// 3. "Click 'Run Ground Truth' to see exact means, then compare estimators"
// Includes a glossary section: wire mean, gate types, MSE, sampling budget.
```

**Step 3: Add PanelHelp to each analytics panel**

Add `<PanelHelp>` with appropriate help text to:
- Gate Structure Analysis
- Wire Mean Distribution
- Error Heatmap
- Estimator Comparison

**Step 4: Show Onboarding in empty state**

When no estimators have been run and circuit exists, show Onboarding instead of empty panels.

**Step 5: Commit**

```bash
git add tools/circuit-explorer/src/components/PanelHelp.jsx tools/circuit-explorer/src/components/Onboarding.jsx tools/circuit-explorer/src/App.jsx tools/circuit-explorer/src/App.css
git commit -m "feat: add in-UI documentation — panel help and onboarding guide"
```

---

### Task 7: Update Slider Ranges and Performance Test

**Files:**
- Modify: `tools/circuit-explorer/src/components/Controls.jsx`

**Step 1: Increase slider ranges**

- Width: max → 1024
- Depth: max → 256

**Step 2: Performance test at scale**

Test these configurations and record time-to-render:
- 64×64 (JointJS mode)
- 128×128 (heatmap mode)
- 512×128 (heatmap mode)
- 1024×256 (heatmap mode, full challenge scale)

Expected: All render within 2 seconds. Hover interactions remain responsive.

**Step 3: Commit**

```bash
git add tools/circuit-explorer/src/components/Controls.jsx
git commit -m "feat: increase slider ranges to 1024 width, 256 depth"
```

---

### Task 8: Clean Up and Remove Old CircuitGraph

**Files:**
- Delete: `tools/circuit-explorer/src/components/CircuitGraph.jsx` (old hand-rolled SVG)
- Modify: `tools/circuit-explorer/src/App.jsx` — remove old import

**Step 1: Remove old component file**

```bash
rm tools/circuit-explorer/src/components/CircuitGraph.jsx
```

**Step 2: Verify build succeeds**

```bash
cd tools/circuit-explorer && npx vite build
```
Expected: Build succeeds with no warnings about missing imports.

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove old CircuitGraph component, replaced by JointJS + Heatmap"
```

---

### Task 9: Final Verification

**Step 1: Full browser test**

Run through the following test matrix:

| Width | Depth | Mode | Test |
|---|---|---|---|
| 8 | 4 | JointJS | Zoom, pan, click gate tooltip |
| 32 | 16 | JointJS | Circuit notation shapes correct |
| 64 | 64 | JointJS | At boundary, still responsive |
| 128 | 64 | Heatmap | Detail overlay hover works |
| 512 | 128 | Heatmap | Canvas renders quickly |
| 1024 | 256 | Heatmap | Full scale, hover overlay |

**Step 2: Run all estimators at 1024×256**

Run Ground Truth, Sampling, Mean Propagation. Verify:
- Timing stats are realistic
- Error heatmap renders
- Analytics panels show meaningful data

**Step 3: Verify build**

```bash
cd tools/circuit-explorer && npx vite build
```
Expected: Clean build, no errors.
