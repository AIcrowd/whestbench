# Network Explorer: MLP Update Design

**Date:** 2026-03-20
**Approach:** Incremental rewrite — replace internals while preserving the React component shell

## Overview

The `tools/network-explorer` visualization tool was built around a Boolean circuit model (gates, wires, bilinear coefficients). The Python backend has been fully refactored to an MLP-based formulation (weight matrices, ReLU activations, Gaussian inputs). This spec covers updating the JS visualization tool to match.

The tool retains its dual-mode structure: **tour mode** (guided pedagogical walkthrough) and **explore mode** (free parameter exploration).

## 1. Data Layer: `circuit.js` → `mlp.js`

Replace `circuit.js` (Boolean gates, bilinear coefficients) with `mlp.js`.

### Public API

```javascript
// MLP generation
sampleMLP(width, depth, seed)
// Returns: { width, depth, weights: Float32Array[] }
// Weights are He-initialized: scale = sqrt(2 / width)

// Simulation (row-vector convention: inputs are rows, matching Python's x @ W)
forwardPass(mlp, inputs)
// inputs: Float32Array of shape (n, width), row-major. Gaussian N(0,1).
// Returns: Array of depth Float32Arrays, each shape (n, width) — one per layer's post-ReLU activations

sampleInputs(n, width, seed)
// Returns: Float32Array(n × width), Gaussian N(0,1) via Box-Muller

outputStats(mlp, nSamples)
// Returns: { means: Float32Array(depth × width), variances: Float32Array(depth × width) }
// Chunked sampling to bound memory, mirrors simulation_fast.py
```

### Implementation details

- Keep the existing seedable PRNG (xoshiro128**) for reproducibility
- Add Box-Muller transform for Gaussian sampling from the uniform PRNG
- Weight generation: `W[i][j] ~ N(0, sqrt(2/width))` per layer
- Forward pass (row-vector convention, matching Python): `x = relu(x @ W)` per layer, where `relu(z) = max(0, z)`. Inputs `x` have shape `(n, width)`, weights `W` have shape `(width, width)`.
- Chunked `outputStats`: process samples in chunks to keep memory O(chunk_size × width) not O(nSamples × width × depth)

### Worker

Rename `circuit.worker.js` → `mlp.worker.js`. Same WebWorker pattern via `useWorker.js` for expensive `outputStats` calls. No changes to worker orchestration.

## 2. Estimators: `estimators.js` Rewrite

Replace bilinear gate math with ReLU moment propagation.

### Mean Propagation

Propagate both E[x] (mean) and Var[x] (diagonal variance) per neuron, assuming independence (E[xy] ≈ E[x]·E[y]).

Per layer, for each neuron `j`:
1. **Pre-activation moments:** `μ_j = Σ_i w_ij · mean_i`, `var_j = Σ_i w_ij² · var_i`
2. **Numerical floor:** `var_j = max(var_j, 1e-12)` to avoid division by zero
3. **ReLU moments:** with `σ_j = sqrt(var_j)` and `α_j = μ_j / σ_j`:
   - `mean_j' = μ_j · Φ(α_j) + σ_j · φ(α_j)`
   - `var_j' = (μ_j² + var_j) · Φ(α_j) + μ_j · σ_j · φ(α_j) - mean_j'²`

Where `Φ` is the standard normal CDF and `φ` is the standard normal PDF.

**Initial layer:** `mean = 0` (Gaussian inputs), `var = 1` for all neurons.

### Covariance Propagation

Track full covariance matrix and mean vector per layer.

Per layer:
1. **Pre-activation mean:** `μ = W^T @ mean` (same as mean propagation)
2. **Pre-activation covariance:** `Cov_pre = W^T @ Cov @ W` (matrix form of `Cov[z_a, z_b] = Σ_ij w_ai · w_bj · Cov[x_i, x_j]`)
3. **Diagonal variances:** `var_j = max(Cov_pre[j,j], 1e-12)`, then `σ_j = sqrt(var_j)`, `α_j = μ_j / σ_j`
4. **Post-ReLU means:** `mean_j' = μ_j · Φ(α_j) + σ_j · φ(α_j)` (same formula as mean propagation)
5. **Post-ReLU covariance (approximate):** `Cov'[a,b] = gain_a · gain_b · Cov_pre[a,b]` where `gain_j = Φ(α_j)`. Then subtract to get centered covariance: `Cov'[a,b] -= mean_a' · mean_b'` (only off-diagonal; diagonal = variance from step 4's extended formula).

**Initial layer:** `mean = 0`, `Cov = I` (identity, since inputs are i.i.d. N(0,1)).

- Storage: flat row-major `Float32Array(n²)` per layer (same pattern as current code)
- Runtime: O(depth × width²) due to matrix multiplications

### Sampling

Generate N input vectors via `sampleInputs`, run `forwardPass`, compute empirical means. Straightforward with the new `mlp.js`.

### Ground Truth

Same as sampling with large N (10,000+), run in WebWorker to avoid blocking UI.

### JS Math Dependencies

Need a normal CDF implementation (no JS stdlib). Options:
- Rational approximation (Abramowitz & Stegun, ~7 significant digits)
- Port from Python `scipy.special.ndtr`

Normal PDF is trivial: `φ(x) = exp(-x²/2) / √(2π)`

### Interface

Each estimator returns a `Float32Array(depth × width)` of predicted neuron means. The `EstimatorRunner.jsx` dropdown and run-button UI remain unchanged.

## 3. Visual Components

### Updated (6 components)

| Component | Changes |
|-----------|---------|
| `CircuitGraphJoint.jsx` | Rewrite to render neuron-and-weight graph for tiny MLPs (width ≤ 8). Neurons as nodes arranged in columns per layer, edges colored/sized by weight value, node color = activation value. |
| `CircuitHeatmap.jsx` → `NetworkHeatmap.jsx` | Rename. Already a (layer × neuron) heatmap. Update labels and color semantics to show neuron activations instead of gate outputs. Primary view for width > 8. |
| `ActivationRibbon.jsx` | Minimal changes — update labels from "gate outputs" to "neuron activations". Data shape (depth, width) is unchanged. |
| `ErrorHeatmap.jsx` / `StdHeatmap.jsx` | Update labels only. Data shape unchanged. |
| `EstimatorComparison.jsx` | Update labels only. Per-layer MSE bar chart logic unchanged. |
| `CoeffHistograms.jsx` | Becomes weight distribution histograms per layer. Demote to collapsible panel. |

### Removed (3 components)

| Component | Reason |
|-----------|--------|
| `GateStats.jsx` | No gate types in MLP |
| `ErrorByGateType.jsx` | No gate types in MLP |
| `WireStats.jsx` | Redundant with ActivationRibbon |

### No new components added.

## 4. Tour Mode (6 Steps)

Tour is locked to **width=8, depth=6** — small enough for the neuron graph, deep enough to show estimator drift. Controls disabled during tour.

| Step | Title | Description | View |
|------|-------|-------------|------|
| 1 | Meet the MLP | Generate small MLP. Show neuron graph with weight connections. Explain: inputs are Gaussian, each layer applies weights + ReLU. | Neuron graph |
| 2 | Watch It Compute | Send a batch of inputs through. Highlight activations flowing layer by layer. Show how ReLU zeros out negative values. | Neuron graph + activation colors |
| 3 | The Goal: Predict Neuron Means | Run ground truth (10k samples). Show per-layer mean heatmap. Explain: "Your job is to predict these values without sampling all 10k times." | Switch to heatmap view |
| 4 | Sampling: The Baseline | Run sampling with small budget (50 samples). Show noisy estimates vs ground truth. Overlay error heatmap. Explain O(1/√k) convergence. | Heatmap + error overlay |
| 5 | Mean Propagation: Fast but Fragile | Run mean propagation. Show it matches at shallow layers but drifts at depth. Explain: "It assumes neurons are independent — they're not." | EstimatorComparison + error heatmap |
| 6 | The Challenge | Show covariance propagation as a teaser — more accurate but expensive. Frame the contest: "Can you beat sampling within a compute budget?" Unlock explore mode. | Full dashboard |

## 5. Explore Mode & App Shell

### Parameter controls

- **Width:** 4–256 (slider)
- **Depth:** 2–32 (slider)
- **Seed:** numeric input
- **Sample budget:** slider

### View switching threshold

- Width ≤ 8 → neuron graph (JointJS)
- Width > 8 → heatmap (`NetworkHeatmap`)

(Replaces current 4096-gate threshold)

### Panel layout

```
┌─────────────────────────────────┬──────────────────────┐
│  Network View                   │  Controls            │
│  (graph or heatmap)             │  width / depth / seed│
│                                 │  Estimator Runner    │
├─────────────────────────────────┤                      │
│  Activation Ribbon              │  Budget slider       │
│  (per-layer distributions)      │                      │
├─────────────────────────────────┼──────────────────────┤
│  Estimator Comparison           │  Weight Distributions│
│  (per-layer MSE bars)           │  (collapsible)       │
├─────────────────────────────────┤                      │
│  Error Heatmap / Std Heatmap    │                      │
└─────────────────────────────────┴──────────────────────┘
```

### Terminology updates (global find-replace in all UI text)

- "circuit" → "network" / "MLP"
- "gate" → "neuron"
- "wire" → "connection" / "weight"
- Window title already updated to "Network Explorer" (commit 11cdacc)

### What stays the same

- Tour/explore mode toggle
- Performance overlay timing display
- WebWorker offloading for heavy computations
- `useWorker.js` hook
- Responsive layout structure

## 6. Files Summary

### New files
- `src/mlp.js` — MLP generation, forward pass, Gaussian sampling, output stats
- `src/mlp.worker.js` — WebWorker wrapper for expensive MLP computations

### Rewritten files
- `src/estimators.js` — ReLU moment propagation (mean prop, cov prop)
- `src/components/CircuitGraphJoint.jsx` — neuron-and-weight graph for width ≤ 8

### Renamed files
- `src/components/CircuitHeatmap.jsx` → `src/components/NetworkHeatmap.jsx`

### Updated files (labels/terminology)
- `src/App.jsx` — parameters, tour steps, terminology
- `src/components/ActivationRibbon.jsx`
- `src/components/ErrorHeatmap.jsx`
- `src/components/StdHeatmap.jsx`
- `src/components/EstimatorComparison.jsx`
- `src/components/CoeffHistograms.jsx` (+ make collapsible)
- `src/components/EstimatorRunner.jsx`
- `src/components/NarrativeCard.jsx` (tour text)
- `src/components/StepIndicator.jsx` (tour step labels)

### Deleted files
- `src/circuit.js`
- `src/circuit.worker.js`
- `src/components/GateStats.jsx`
- `src/components/ErrorByGateType.jsx`
- `src/components/WireStats.jsx`
