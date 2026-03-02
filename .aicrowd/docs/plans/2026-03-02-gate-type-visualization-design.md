# Gate Type Visualization & Error Analysis

## Background

The ARC reference implementation generates circuits whose gates are genuine boolean functions expressed as multilinear polynomials over {-1, +1} inputs. Our JS `circuit.js` generates the same distribution of gate types. Currently the circuit view renders all gates as identical rectangles — this design adds gate type awareness to unlock visual identity and per-type error analysis.

## Gate Type Taxonomy

Every gate computes `out = c + a·x + b·y + p·x·y`. The coefficients uniquely determine the boolean function:

| Category | Gate | Coefficients (c, a, b, p) | Label in UI |
|---|---|---|---|
| Simple | Buffer | (0, ±1, 0, 0) or (0, 0, ±1, 0) | `BUF ▷` |
| Simple | NOT | (0, ∓1, 0, 0) or (0, 0, ∓1, 0) | `NOT ▷○` |
| Simple | Constant | (±1, 0, 0, 0) | `±1` |
| Simple | XOR | (0, 0, 0, -1) | `XOR ⊕` |
| Simple | XNOR | (0, 0, 0, +1) | `XNOR ⊙` |
| Complex | AND | (-0.5, 0.5, 0.5, 0.5) | `AND ∧` |
| Complex | NAND | (0.5, -0.5, -0.5, 0.5) | `NAND ∧̄` |
| Complex | OR | (-0.5, 0.5, 0.5, -0.5) | `OR ∨` |
| Complex | NOR | (0.5, -0.5, -0.5, -0.5) | `NOR ∨̄` |

Input negation variants (e.g., AND(-x, y)) are shown in the label text rather than via port decorations.

## Feature 1: Gate Type Metadata

Extend `classifyGate()` in `gateShapes.js` from the current 4-category system (`linear`, `constant`, `product`, `and`) to a full boolean function classifier that returns:

```js
{ type: "AND", label: "AND(-x, y)", symbol: "∧" }
```

The `describeGate()` function in `circuit.js` already has similar logic and can be consolidated.

## Feature 2: Gate Symbol Rendering (Small Circuit Mode)

In `CircuitGraphJoint.jsx`, render the gate type symbol as an SVG text element inside each rectangle. The existing rectangle shape, E[wire] color-coding on output ports, and tooltip system remain unchanged.

- **Inside rectangle**: Gate symbol (e.g., `∧`, `⊕`, `▷`) as centered SVG text
- **Tooltip**: Add gate type row (e.g., "AND(-x, y)") to the existing coefficient display
- **Legend**: A row below the circuit header showing all gate type symbols with their names, consistent with the bar chart legend

## Feature 3: Error by Gate Type Chart

New `ErrorByGateType` component:

- **X-axis**: Gate type categories (AND, OR, XOR, NOT, BUF, CONST, etc.)
- **Y-axis**: Mean |error| for gates of that type at the active layer
- **Bars**: Grouped by estimator (sampling, mean prop, cov prop) using `SERIES_COLORS`
- **Integration**: Uses `activeLayer` from App state; requires ground truth + estimator results + gate type metadata

## Files to Modify

### `gateShapes.js`
- Expand `classifyGate()` to return full boolean gate type with label and symbol
- Add gate type color palette and symbol constants
- Consolidate with `describeGate()` from `circuit.js`

### `CircuitGraphJoint.jsx`
- Add SVG text label (symbol) inside each gate rectangle
- Add gate type to `gateData` stored on each node
- Add gate type row to tooltip
- Add legend row below header

### `circuit.js`
- Remove `describeGate()` (moved to `gateShapes.js`)

### New: `ErrorByGateType.jsx`
- Canvas-rendered grouped bar chart
- Groups gates by type per active layer
- Computes mean |error| per gate type per estimator

### `App.jsx`
- Pass gate type data + estimator results to new component
- Add component to dashboard grid

## Verification Plan

### Manual Verification
1. Generate small circuits (3-5 wires, 3-5 layers) and visually verify:
   - Each gate rectangle shows the correct symbol
   - Tooltip shows correct gate type label (e.g., "AND(-x, y)")
   - Legend matches the symbols shown in the circuit
2. Click gates and verify tooltip gate type matches the coefficients displayed
3. Generate a circuit and verify the ErrorByGateType chart:
   - Shows bars for each gate type present in the active layer
   - Bars change when switching layers
   - Bar values are reasonable (higher error for AND/OR types in mean propagation)
4. Resize to trigger heatmap mode and verify symbols are NOT shown in heatmap mode
