# Scalable Circuit Visualization — Design

## Problem

The Circuit Explorer currently renders each wire as an SVG circle and draws all connections. This works at 8–64 wires but collapses at 1024 wires × 256 layers (262k nodes, 524k edges). Participants need to explore circuits at the challenge's full scale.

## Approach: Adaptive Dual-Mode

The mode switch is based on **total gate count** (width × depth), not just width, since depth also impacts rendering performance.

| Total Gates (n × d) | Example Sizes | Circuit View | Analytics |
|---|---|---|---|
| ≤ 4,096 | 64×64, 32×128, 16×256 | **JointJS** interactive graph — zoom, pan, click gates | Recharts panels |
| > 4,096 | 128×64, 256×32, 1024×256 | **Heatmap** (wires × layers) with hover-to-trace | Recharts panels |

At depth 256, JointJS is used up to ~16 wires. At depth 32, JointJS handles up to 128 wires. The heatmap is always available as a secondary view for any size.

---

## Mode 1: JointJS Graph (≤ 64 wires)

Replaces the current hand-rolled SVG node graph.

**Library**: `@joint/core` (open source, MIT-like license)

**What it provides**:
- SVG-based rendering (crisp at any zoom)
- Native zoom/pan via `paper.scale()` and scroll events
- Click-to-inspect: clicking a gate shows its operation, coefficients, inputs
- Proper layout: gates arranged in columns per layer, wires as links

**Circuit-style gate rendering**:

Every gate in this circuit computes the same parametric form:
```
output = c + a·x + b·y + p·x·y
```
where `x = wire[first]`, `y = wire[second]`, and the four coefficients `(c, a, b, p)` determine behavior. There are no distinct AND/OR/etc. gate types — just coefficient patterns. The generation produces two categories:

**Simple gates** (one nonzero coeff):
| Dominant Coeff | Meaning | Visual Shape | Label |
|---|---|---|---|
| `a ≠ 0` only | passthrough x | **triangle** (buffer) | `x` or `-x` |
| `b ≠ 0` only | passthrough y | **triangle** (buffer) | `y` or `-y` |
| `c ≠ 0` only | constant | **small square** with value | `+1` or `-1` |
| `p ≠ 0` only | pure product | **circle** with × | `xy` or `-xy` |

**Complex gates** (all four coeffs, AND-like):
| Pattern | Visual Shape | Label |
|---|---|---|
| `c, a, b, p` all set | **D-shape** (AND-gate style) | `AND(±x, ±y)` |

Each gate node also shows:
- Its two input wire indices as port labels
- The coefficient values on hover/click
- Color fill based on wire mean when estimator results exist (blue → white → red)

Wire connections use right-angle routing (circuit-board style) connecting output ports to input ports of downstream gates.

**Estimated element count at max (64 × 32)**:
- 2,048 nodes + ~4,096 links = ~6k SVG elements → comfortably within JointJS performance

**Interaction**:
- Click gate → tooltip showing: `gate[l][w]: AND(-x₃, x₅) → mean = 0.142`
- Zoom/pan with mouse wheel and drag
- Layer highlighting: hover a layer column to dim other layers

---

## Mode 2: Heatmap Dashboard (> 64 wires)

For large circuits, individual nodes are not useful. The circuit is visualized as a 2D heatmap grid.

**Primary panel — Wire Means Heatmap** (already exists, enhanced):
- X-axis = wires (0..n-1), Y-axis = layers (0..d-1)
- Color = wire mean value (blue → white → red diverging scale)
- At 1024×256, this is a dense but readable image

**Hover detail overlay** (input band → gate → output band):

When hovering a cell at (wire `w`, layer `l`), a detail overlay appears showing the gate's local neighborhood:

```
┌──────┐  ┌─────┐  ┌──────┐
│Layer │  │Gate │  │Layer │
│ l-1  │  │ [w] │  │ l+1  │
│      │  │     │  │      │
│ ███  │  │ c+  │  │      │
│ ▶██  │──│ax + │──│ ██◀  │
│ ▶██  │  │by + │  │ ██◀  │
│ ███  │  │pxy  │  │ ██◀  │
│ ███  │  │     │  │ ███  │
└──────┘  └─────┘  └──────┘
```

- **Left band**: 1D vertical heatmap of all wire means at layer `l−1`. The two input wires (`first[w]`, `second[w]`) are accented with markers (▶) and brighter color.
- **Center**: Gate element showing the operation (`AND(−x₃, x₅)`), coefficients `(c, a, b, p)`, and the output wire mean.
- **Right band**: 1D vertical heatmap of all wire means at layer `l+1`. Wires that consume wire `w` as input are accented (◀).

This works at any scale — even at 1024 wires, the bands show the full signal distribution as a continuous color gradient with connected wires popping out as distinct markers.

---

## Analytics Dashboard (both modes)

These panels work at any circuit size since they aggregate per layer:

### Existing panels (keep):
- **Gate Structure Analysis** — stacked bar of gate types per layer + coefficient magnitudes
- **Wire Mean Distribution** — μ ± σ band chart per layer + scatter of individual wire means
- **Estimation Error (MSE)** — per-layer MSE comparing estimators

### New panel — Error Heatmap:
- Same layout as Wire Means Heatmap but showing `|ground_truth[l][w] - estimate[l][w]|`
- Immediately reveals which wires/layers the estimator struggles with
- Most valuable diagnostic for participants debugging their estimators

### Enhanced panel — Estimator Comparison:
- Side-by-side wire means heatmaps for GT vs Sampling vs Mean Prop
- Or a single heatmap with a dropdown toggle between views

---

## Dashboard Layout

```
┌──────────────────────────────────────────┐
│ Header                                    │
├─────────┬────────────────────────────────┤
│ Sidebar │  Circuit View                  │
│ - Params│  (JointJS graph OR heatmap)    │
│ - Run   │                                │
│ - Results│                               │
│         ├────────────┬───────────────────┤
│         │ Gate Stats │ Wire Distribution │
│         ├────────────┴───────────────────┤
│         │ Error Heatmap (new)            │
│         ├────────────────────────────────┤
│         │ Estimator Comparison           │
└─────────┴────────────────────────────────┘
```

---

## In-UI Documentation

The interface should be self-explanatory for a new user encountering the challenge for the first time.

**Panel-level help**:
- Each panel has an ℹ️ icon that expands a short explanation
- Example: Gate Structure panel → "Shows what kind of operations dominate each layer. Product-heavy layers are harder to estimate because E[xy] ≠ E[x]·E[y] when inputs are correlated."
- Example: Error Heatmap → "Bright cells = high estimation error. Look for patterns: does error grow with depth? Do certain wires consistently fail?"

**Guided onboarding** (first-time state):
- When no estimator has been run, the main area shows a brief guide:
  1. "This circuit has N wires and D layers of random gates"
  2. "Your goal: estimate the mean output of each wire without running all 2^N inputs"
  3. "Click 'Run Ground Truth' to see exact means, then try Sampling and Mean Propagation to compare"

**Glossary sidebar** (collapsible):
- Wire mean: E[f(x)] over uniform ±1 inputs
- Gate types: constant, linear, product, AND
- MSE: mean squared error between ground truth and estimate
- Sampling budget: number of random ±1 inputs evaluated

---

## Dependencies

| Package | Purpose | Size |
|---|---|---|
| `@joint/core` | Interactive circuit graph | ~300KB min |
| `recharts` | Analytics charts | Already installed |

---

## Verification Plan

1. Test JointJS graph at 8, 32, 64 wires — verify zoom/pan, gate inspection
2. Test heatmap mode at 128, 512, 1024 wires — verify rendering performance
3. Test auto-switching at boundary (64 → 65 wires)
4. Verify hover-to-trace connections work on heatmap
5. Verify error heatmap shows meaningful patterns
