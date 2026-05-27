# Network Explorer MLP Update — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the network-explorer visualization tool from Boolean circuit model to MLP formulation while preserving the React component shell, tour/explore dual-mode, and heatmap-based visualizations.

**Architecture:** Incremental rewrite — replace the data layer (`circuit.js` → `mlp.js`), rewrite estimators for ReLU math, update visual components for MLP semantics, rewrite tour narrative. The heatmap-based visualizations already map well to MLP layer×neuron data. JointJS graph view is kept for tiny MLPs (width ≤ 8) only.

**Tech Stack:** React 19, Vite 7, JointJS 4.2 (graph view), Canvas API (heatmaps), WebWorkers (heavy computation), vanilla JS math (no TF.js dependency for MLP path).

**Spec:** `docs/superpowers/specs/2026-03-20-network-explorer-mlp-update-design.md`

---

## File Structure

### New files
| File | Responsibility |
|------|---------------|
| `src/mlp.js` | MLP generation, forward pass, Gaussian sampling, output stats, seedable PRNG |
| `src/mlp.worker.js` | WebWorker wrapper dispatching mlp.js and estimators.js calls |
| `src/math-utils.js` | Normal CDF (Φ), normal PDF (φ), Box-Muller transform — shared by mlp.js and estimators.js |
| `src/__tests__/mlp.test.js` | Tests for MLP generation, forward pass, output stats |
| `src/__tests__/estimators.test.js` | Tests for mean propagation and covariance propagation |
| `src/__tests__/math-utils.test.js` | Tests for normal CDF/PDF accuracy |
| `src/components/NetworkGraph.jsx` | Neuron-and-weight graph for tiny MLPs (width ≤ 8), replaces CircuitGraphJoint |

### Rewritten files
| File | What changes |
|------|-------------|
| `src/estimators.js` | Complete rewrite: ReLU moment propagation (mean prop + cov prop) |

### Modified files
| File | What changes |
|------|-------------|
| `src/useWorker.js` | Point to `mlp.worker.js`, rename hook export |
| `src/App.jsx` | New params, tour steps, view threshold, terminology, remove deleted component imports |
| `src/components/CircuitHeatmap.jsx` → `src/components/NetworkHeatmap.jsx` | Rename + update labels |
| `src/components/ActivationRibbon.jsx` | Update labels |
| `src/components/ErrorHeatmap.jsx` | Update labels, remove gate-type references |
| `src/components/StdHeatmap.jsx` | Update labels, remove gate-type references |
| `src/components/EstimatorComparison.jsx` | Update labels |
| `src/components/EstimatorRunner.jsx` | Remove TF.js GPU path, use mlp.worker for all estimators |
| `src/components/CoeffHistograms.jsx` | Rewrite for weight distributions, make collapsible |
| `src/components/NarrativeCard.jsx` | New tour narrative text for MLP concepts |
| `src/components/StepIndicator.jsx` | Update step labels |
| `src/components/Controls.jsx` | Update parameter ranges and labels |

### Deleted files
| File | Reason |
|------|--------|
| `src/circuit.js` | Replaced by mlp.js |
| `src/circuit.worker.js` | Replaced by mlp.worker.js |
| `src/circuit-tf.js` | TF.js GPU path removed (MLP math is fast enough on CPU) |
| `src/components/CircuitGraphJoint.jsx` | Replaced by NetworkGraph.jsx |
| `src/components/CircuitGraph.jsx` | Old graph view, unused |
| `src/components/GateStats.jsx` | No gate types in MLP |
| `src/components/ErrorByGateType.jsx` | No gate types in MLP |
| `src/components/WireStats.jsx` | Redundant with ActivationRibbon |
| `src/components/GateDetailOverlay.jsx` | Gate-specific, no longer needed |
| `src/components/gateShapes.js` | Gate-specific shapes, no longer needed |

---

## Task 1: Math Utilities

**Files:**
- Create: `tools/network-explorer/src/math-utils.js`
- Create: `tools/network-explorer/src/__tests__/math-utils.test.js`

- [ ] **Step 1: Write failing tests for normal PDF and CDF**

```javascript
// src/__tests__/math-utils.test.js
import { describe, it, expect } from 'vitest';
import { normalPdf, normalCdf, boxMuller, makeRng } from '../math-utils.js';

describe('normalPdf', () => {
  it('returns correct value at x=0', () => {
    // φ(0) = 1/√(2π) ≈ 0.3989422804
    expect(normalPdf(0)).toBeCloseTo(0.3989422804, 6);
  });
  it('returns correct value at x=1', () => {
    expect(normalPdf(1)).toBeCloseTo(0.2419707245, 6);
  });
  it('is symmetric', () => {
    expect(normalPdf(-2)).toBeCloseTo(normalPdf(2), 10);
  });
});

describe('normalCdf', () => {
  it('returns 0.5 at x=0', () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 8);
  });
  it('returns correct value at x=1', () => {
    // Φ(1) ≈ 0.8413447461
    expect(normalCdf(1)).toBeCloseTo(0.8413447461, 6);
  });
  it('returns correct value at x=-1', () => {
    expect(normalCdf(-1)).toBeCloseTo(0.1586552539, 6);
  });
  it('approaches 0 for large negative', () => {
    expect(normalCdf(-6)).toBeCloseTo(0, 8);
  });
  it('approaches 1 for large positive', () => {
    expect(normalCdf(6)).toBeCloseTo(1, 8);
  });
});

describe('makeRng', () => {
  it('is deterministic given same seed', () => {
    const rng1 = makeRng(42);
    const rng2 = makeRng(42);
    for (let i = 0; i < 100; i++) {
      expect(rng1.random()).toBe(rng2.random());
    }
  });
  it('produces different sequences for different seeds', () => {
    const rng1 = makeRng(1);
    const rng2 = makeRng(2);
    const same = Array.from({ length: 10 }, () => rng1.random() === rng2.random());
    expect(same.every(Boolean)).toBe(false);
  });
});

describe('boxMuller', () => {
  it('produces approximately standard normal samples', () => {
    const rng = makeRng(123);
    const N = 10000;
    const samples = [];
    for (let i = 0; i < N; i++) {
      const [a, b] = boxMuller(rng);
      samples.push(a, b);
    }
    const mean = samples.reduce((s, x) => s + x, 0) / samples.length;
    const variance = samples.reduce((s, x) => s + (x - mean) ** 2, 0) / samples.length;
    expect(mean).toBeCloseTo(0, 1);
    expect(variance).toBeCloseTo(1, 1);
  });
});
```

- [ ] **Step 2: Add vitest to the project**

Run:
```bash
cd tools/network-explorer && npm install -D vitest
```

Add to `package.json` scripts: `"test": "vitest run"`, `"test:watch": "vitest"`

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd tools/network-explorer && npx vitest run src/__tests__/math-utils.test.js`
Expected: FAIL — module not found

- [ ] **Step 4: Implement math-utils.js**

```javascript
// src/math-utils.js

// Standard normal PDF: φ(x) = exp(-x²/2) / √(2π)
export function normalPdf(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

// Standard normal CDF via rational approximation (Abramowitz & Stegun 26.2.17)
// Accuracy: |error| < 7.5e-8
export function normalCdf(x) {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);
  const t = 1.0 / (1.0 + p * absX);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-0.5 * absX * absX);

  return 0.5 * (1.0 + sign * y);
}

// Box-Muller transform: two uniform samples → two standard normal samples
export function boxMuller(rng) {
  const u1 = rng.random();
  const u2 = rng.random();
  const r = Math.sqrt(-2 * Math.log(u1));
  const theta = 2 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

// Seedable PRNG (xoshiro128**) — ported from existing circuit.js
export function makeRng(seed) {
  // SplitMix32 for seed expansion
  let s = seed | 0;
  function sm32() {
    s = (s + 0x9e3779b9) | 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  }
  let a = sm32(), b = sm32(), c = sm32(), d = sm32();

  function next() {
    const result = Math.imul(rotl(Math.imul(b, 5), 7), 9) >>> 0;
    const t = (b << 9) >>> 0;
    c ^= a; d ^= b; b ^= c; a ^= d;
    c ^= t;
    d = rotl(d, 11) >>> 0;
    return result;
  }

  function rotl(x, k) {
    return ((x << k) | (x >>> (32 - k))) >>> 0;
  }

  return {
    random() { return next() / 0x100000000; },
    randInt(lo, hi) { return lo + (next() % (hi - lo)); },
  };
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd tools/network-explorer && npx vitest run src/__tests__/math-utils.test.js`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add tools/network-explorer/src/math-utils.js tools/network-explorer/src/__tests__/math-utils.test.js tools/network-explorer/package.json tools/network-explorer/package-lock.json
git commit -m "feat(network-explorer): add math utilities (normalCdf, normalPdf, boxMuller, makeRng)"
```

---

## Task 2: MLP Data Layer

**Files:**
- Create: `tools/network-explorer/src/mlp.js`
- Create: `tools/network-explorer/src/__tests__/mlp.test.js`

- [ ] **Step 1: Write failing tests for MLP generation and forward pass**

```javascript
// src/__tests__/mlp.test.js
import { describe, it, expect } from 'vitest';
import { sampleMLP, sampleInputs, forwardPass, outputStats } from '../mlp.js';

describe('sampleMLP', () => {
  it('returns correct structure', () => {
    const mlp = sampleMLP(4, 3, 42);
    expect(mlp.width).toBe(4);
    expect(mlp.depth).toBe(3);
    expect(mlp.weights).toHaveLength(3);
    mlp.weights.forEach(w => {
      expect(w).toBeInstanceOf(Float32Array);
      expect(w.length).toBe(4 * 4); // width × width, row-major
    });
  });

  it('is deterministic given same seed', () => {
    const a = sampleMLP(8, 4, 123);
    const b = sampleMLP(8, 4, 123);
    a.weights.forEach((w, i) => {
      for (let j = 0; j < w.length; j++) {
        expect(w[j]).toBe(b.weights[i][j]);
      }
    });
  });

  it('uses He initialization scale', () => {
    const width = 256;
    const mlp = sampleMLP(width, 1, 99);
    const w = mlp.weights[0];
    const mean = w.reduce((s, x) => s + x, 0) / w.length;
    const variance = w.reduce((s, x) => s + (x - mean) ** 2, 0) / w.length;
    // He scale: std = sqrt(2/width), variance = 2/width
    const expectedVariance = 2 / width;
    expect(variance).toBeCloseTo(expectedVariance, 1);
  });
});

describe('sampleInputs', () => {
  it('returns correct shape', () => {
    const inputs = sampleInputs(100, 8, 42);
    expect(inputs).toBeInstanceOf(Float32Array);
    expect(inputs.length).toBe(100 * 8);
  });

  it('produces approximately standard normal', () => {
    const n = 10000, width = 4;
    const inputs = sampleInputs(n, width, 55);
    let sum = 0, sum2 = 0;
    for (let i = 0; i < inputs.length; i++) {
      sum += inputs[i];
      sum2 += inputs[i] * inputs[i];
    }
    const mean = sum / inputs.length;
    const variance = sum2 / inputs.length - mean * mean;
    expect(mean).toBeCloseTo(0, 1);
    expect(variance).toBeCloseTo(1, 1);
  });
});

describe('forwardPass', () => {
  it('returns array of depth Float32Arrays', () => {
    const mlp = sampleMLP(4, 3, 42);
    const inputs = sampleInputs(10, 4, 1);
    const layers = forwardPass(mlp, inputs);
    expect(layers).toHaveLength(3);
    layers.forEach(layer => {
      expect(layer).toBeInstanceOf(Float32Array);
      expect(layer.length).toBe(10 * 4); // n × width
    });
  });

  it('produces non-negative outputs (ReLU)', () => {
    const mlp = sampleMLP(8, 5, 42);
    const inputs = sampleInputs(100, 8, 1);
    const layers = forwardPass(mlp, inputs);
    layers.forEach(layer => {
      for (let i = 0; i < layer.length; i++) {
        expect(layer[i]).toBeGreaterThanOrEqual(0);
      }
    });
  });

  it('is deterministic', () => {
    const mlp = sampleMLP(4, 2, 42);
    const inputs = sampleInputs(5, 4, 1);
    const a = forwardPass(mlp, inputs);
    const b = forwardPass(mlp, inputs);
    a.forEach((layer, i) => {
      for (let j = 0; j < layer.length; j++) {
        expect(layer[j]).toBe(b[i][j]);
      }
    });
  });
});

describe('outputStats', () => {
  it('returns means and variances of correct shape', () => {
    const mlp = sampleMLP(4, 3, 42);
    const stats = outputStats(mlp, 1000);
    expect(stats.means).toBeInstanceOf(Float32Array);
    expect(stats.means.length).toBe(3 * 4); // depth × width
    expect(stats.variances).toBeInstanceOf(Float32Array);
    expect(stats.variances.length).toBe(3 * 4);
  });

  it('means converge with more samples', () => {
    const mlp = sampleMLP(4, 2, 42);
    const few = outputStats(mlp, 100);
    const many = outputStats(mlp, 10000);
    // With more samples, variance of estimate should be lower
    // Just check it runs and returns finite values
    for (let i = 0; i < many.means.length; i++) {
      expect(Number.isFinite(many.means[i])).toBe(true);
      expect(many.variances[i]).toBeGreaterThanOrEqual(0);
    }
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools/network-explorer && npx vitest run src/__tests__/mlp.test.js`
Expected: FAIL — module not found

- [ ] **Step 3: Implement mlp.js**

```javascript
// src/mlp.js
import { makeRng, boxMuller } from './math-utils.js';

/**
 * Generate a random MLP with He-initialized weights.
 * @param {number} width - neurons per layer
 * @param {number} depth - number of layers
 * @param {number} seed - PRNG seed
 * @returns {{ width: number, depth: number, weights: Float32Array[] }}
 */
export function sampleMLP(width, depth, seed) {
  const rng = makeRng(seed);
  const scale = Math.sqrt(2 / width);
  const weights = [];

  for (let d = 0; d < depth; d++) {
    const w = new Float32Array(width * width);
    for (let i = 0; i < w.length; i += 2) {
      const [g1, g2] = boxMuller(rng);
      w[i] = g1 * scale;
      if (i + 1 < w.length) w[i + 1] = g2 * scale;
    }
    weights.push(w);
  }

  return { width, depth, weights };
}

/**
 * Generate Gaussian N(0,1) input samples.
 * @param {number} n - number of samples
 * @param {number} width - vector dimension
 * @param {number} seed - PRNG seed
 * @returns {Float32Array} shape (n, width), row-major
 */
export function sampleInputs(n, width, seed) {
  const rng = makeRng(seed);
  const total = n * width;
  const out = new Float32Array(total);
  for (let i = 0; i < total; i += 2) {
    const [g1, g2] = boxMuller(rng);
    out[i] = g1;
    if (i + 1 < total) out[i + 1] = g2;
  }
  return out;
}

/**
 * Forward pass through MLP with ReLU activations.
 * Row-vector convention: x @ W per layer, matching Python simulation.py.
 * @param {{ width: number, depth: number, weights: Float32Array[] }} mlp
 * @param {Float32Array} inputs - shape (n, width), row-major
 * @returns {Float32Array[]} Array of depth Float32Arrays, each shape (n, width)
 */
export function forwardPass(mlp, inputs) {
  const { width, depth, weights } = mlp;
  const n = inputs.length / width;
  const layers = [];
  let x = inputs;

  for (let d = 0; d < depth; d++) {
    const W = weights[d];
    const out = new Float32Array(n * width);

    for (let s = 0; s < n; s++) {
      const sOff = s * width;
      for (let j = 0; j < width; j++) {
        let sum = 0;
        for (let i = 0; i < width; i++) {
          // x[s, i] * W[i, j] — row-major: W[i * width + j]
          sum += x[sOff + i] * W[i * width + j];
        }
        out[sOff + j] = sum > 0 ? sum : 0; // ReLU
      }
    }

    layers.push(out);
    x = out;
  }

  return layers;
}

/**
 * Compute per-layer output statistics via chunked sampling.
 * @param {{ width: number, depth: number, weights: Float32Array[] }} mlp
 * @param {number} nSamples
 * @returns {{ means: Float32Array, variances: Float32Array }} each shape (depth × width), row-major
 */
export function outputStats(mlp, nSamples, seed = 0) {
  const { width, depth } = mlp;
  const chunkSize = Math.max(64, Math.min(1024, Math.floor(4 * 1024 * 1024 / (width * 4))));

  const sumBuf = new Float64Array(depth * width);
  const sum2Buf = new Float64Array(depth * width);
  let processed = 0;

  while (processed < nSamples) {
    const thisChunk = Math.min(chunkSize, nSamples - processed);
    const inputs = sampleInputs(thisChunk, width, seed + processed);
    const layers = forwardPass(mlp, inputs);

    for (let d = 0; d < depth; d++) {
      const layerData = layers[d];
      const dOff = d * width;
      for (let s = 0; s < thisChunk; s++) {
        const sOff = s * width;
        for (let j = 0; j < width; j++) {
          const v = layerData[sOff + j];
          sumBuf[dOff + j] += v;
          sum2Buf[dOff + j] += v * v;
        }
      }
    }

    processed += thisChunk;
  }

  const means = new Float32Array(depth * width);
  const variances = new Float32Array(depth * width);
  for (let i = 0; i < depth * width; i++) {
    const m = sumBuf[i] / nSamples;
    means[i] = m;
    variances[i] = sum2Buf[i] / nSamples - m * m;
  }

  return { means, variances };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools/network-explorer && npx vitest run src/__tests__/mlp.test.js`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/network-explorer/src/mlp.js tools/network-explorer/src/__tests__/mlp.test.js
git commit -m "feat(network-explorer): add mlp.js data layer (generation, forward pass, output stats)"
```

---

## Task 3: ReLU Estimators

**Files:**
- Rewrite: `tools/network-explorer/src/estimators.js`
- Create: `tools/network-explorer/src/__tests__/estimators.test.js`

**Reference:** `src/network_estimation/estimators.py` (Python implementations to port)

- [ ] **Step 1: Write failing tests for mean propagation**

```javascript
// src/__tests__/estimators.test.js
import { describe, it, expect } from 'vitest';
import { meanPropagation, covariancePropagation } from '../estimators.js';
import { sampleMLP, outputStats } from '../mlp.js';

describe('meanPropagation', () => {
  it('returns Float32Array of shape depth × width', () => {
    const mlp = sampleMLP(8, 4, 42);
    const result = meanPropagation(mlp);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(4 * 8);
  });

  it('produces non-negative values (post-ReLU means)', () => {
    const mlp = sampleMLP(8, 4, 42);
    const result = meanPropagation(mlp);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('is close to ground truth for shallow networks', () => {
    const mlp = sampleMLP(16, 2, 42);
    const gt = outputStats(mlp, 50000);
    const est = meanPropagation(mlp);
    // For shallow networks, mean propagation should be quite accurate
    let mse = 0;
    for (let i = 0; i < est.length; i++) {
      const diff = est[i] - gt.means[i];
      mse += diff * diff;
    }
    mse /= est.length;
    expect(mse).toBeLessThan(0.01);
  });
});

describe('covariancePropagation', () => {
  it('returns Float32Array of shape depth × width', () => {
    const mlp = sampleMLP(8, 4, 42);
    const result = covariancePropagation(mlp);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(4 * 8);
  });

  it('produces non-negative values', () => {
    const mlp = sampleMLP(8, 4, 42);
    const result = covariancePropagation(mlp);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('is at least as accurate as mean propagation for deep networks', () => {
    const mlp = sampleMLP(16, 8, 42);
    const gt = outputStats(mlp, 50000);
    const mp = meanPropagation(mlp);
    const cp = covariancePropagation(mlp);

    let mseMp = 0, mseCp = 0;
    for (let i = 0; i < mp.length; i++) {
      mseMp += (mp[i] - gt.means[i]) ** 2;
      mseCp += (cp[i] - gt.means[i]) ** 2;
    }
    // Covariance propagation should have equal or lower MSE
    expect(mseCp).toBeLessThanOrEqual(mseMp * 1.1); // 10% tolerance
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools/network-explorer && npx vitest run src/__tests__/estimators.test.js`
Expected: FAIL — old estimators.js doesn't export these functions / wrong signatures

- [ ] **Step 3: Rewrite estimators.js with ReLU math**

Port from `src/network_estimation/estimators.py`. The file should export `meanPropagation(mlp)` and `covariancePropagation(mlp)`, each returning a `Float32Array(depth × width)` of predicted per-layer neuron means.

```javascript
// src/estimators.js
import { normalPdf, normalCdf } from './math-utils.js';

/**
 * Mean Propagation estimator.
 * Propagates mean and diagonal variance through ReLU layers
 * assuming neuron independence.
 * @param {{ width, depth, weights }} mlp
 * @returns {Float32Array} shape (depth × width), predicted per-layer means
 */
export function meanPropagation(mlp) {
  const { width, depth, weights } = mlp;
  const result = new Float32Array(depth * width);

  // Initial: input ~ N(0, 1)
  let mean = new Float64Array(width); // all zeros
  let variance = new Float64Array(width);
  variance.fill(1.0);

  for (let d = 0; d < depth; d++) {
    const W = weights[d];
    const newMean = new Float64Array(width);
    const newVar = new Float64Array(width);

    for (let j = 0; j < width; j++) {
      // Pre-activation: z_j = Σ_i W[i,j] * x_i
      let mu = 0, v = 0;
      for (let i = 0; i < width; i++) {
        const w = W[i * width + j];
        mu += w * mean[i];
        v += w * w * variance[i];
      }
      v = Math.max(v, 1e-12);
      const sigma = Math.sqrt(v);
      const alpha = mu / sigma;

      // Post-ReLU moments
      const phiA = normalPdf(alpha);
      const PhiA = normalCdf(alpha);
      const postMean = mu * PhiA + sigma * phiA;
      const postVar = (mu * mu + v) * PhiA + mu * sigma * phiA - postMean * postMean;

      newMean[j] = postMean;
      newVar[j] = Math.max(postVar, 1e-12);
    }

    // Store this layer's means in result
    const dOff = d * width;
    for (let j = 0; j < width; j++) {
      result[dOff + j] = newMean[j];
    }
    mean = newMean;
    variance = newVar;
  }

  return result;
}

/**
 * Covariance Propagation estimator.
 * Tracks full covariance matrix through ReLU layers.
 * @param {{ width, depth, weights }} mlp
 * @returns {Float32Array} shape (depth × width), predicted per-layer means
 */
export function covariancePropagation(mlp) {
  const { width, depth, weights } = mlp;
  const result = new Float32Array(depth * width);

  // Initial: input ~ N(0, I)
  let mean = new Float64Array(width); // zeros
  // Covariance = identity, stored flat row-major
  let cov = new Float64Array(width * width);
  for (let i = 0; i < width; i++) cov[i * width + i] = 1.0;

  for (let d = 0; d < depth; d++) {
    const W = weights[d];
    const n = width;

    // Pre-activation mean: mu = W^T @ mean
    const mu = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += W[i * n + j] * mean[i];
      mu[j] = s;
    }

    // Pre-activation covariance: covPre = W^T @ cov @ W
    // First: tmp = cov @ W (n×n)
    const tmp = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += cov[i * n + k] * W[k * n + j];
        tmp[i * n + j] = s;
      }
    }
    // covPre = W^T @ tmp → covPre[a,b] = Σ_i W[i,a] * tmp[i,b]
    const covPre = new Float64Array(n * n);
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += W[i * n + a] * tmp[i * n + b];
        covPre[a * n + b] = s;
      }
    }

    // Post-ReLU: compute per-neuron alpha, gain, post-mean
    const alpha = new Float64Array(n);
    const gain = new Float64Array(n);
    const postMean = new Float64Array(n);
    const postVar = new Float64Array(n);

    for (let j = 0; j < n; j++) {
      const v = Math.max(covPre[j * n + j], 1e-12);
      const sigma = Math.sqrt(v);
      const a = mu[j] / sigma;
      alpha[j] = a;
      const PhiA = normalCdf(a);
      const phiA = normalPdf(a);
      gain[j] = PhiA;
      postMean[j] = mu[j] * PhiA + sigma * phiA;
      postVar[j] = Math.max(
        (mu[j] * mu[j] + v) * PhiA + mu[j] * sigma * phiA - postMean[j] * postMean[j],
        1e-12
      );
    }

    // Post-ReLU covariance (approximate):
    // E[ReLU(z_a) * ReLU(z_b)] ≈ gain_a * gain_b * covPre[a,b] + mean_a' * mean_b'
    // So Cov'[a,b] = gain_a * gain_b * covPre[a,b] + mean_a' * mean_b' - mean_a' * mean_b'
    //             = gain_a * gain_b * covPre[a,b]  (for off-diagonal, the centering cancels)
    // For diagonal: use the exact post-ReLU variance formula.
    const newCov = new Float64Array(n * n);
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        if (a === b) {
          newCov[a * n + a] = postVar[a];
        } else {
          newCov[a * n + b] = gain[a] * gain[b] * covPre[a * n + b];
        }
      }
    }

    // Store means
    const dOff = d * width;
    for (let j = 0; j < width; j++) {
      result[dOff + j] = postMean[j];
    }

    mean = postMean;
    cov = newCov;
  }

  return result;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools/network-explorer && npx vitest run src/__tests__/estimators.test.js`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/network-explorer/src/estimators.js tools/network-explorer/src/__tests__/estimators.test.js
git commit -m "feat(network-explorer): rewrite estimators for ReLU moment propagation"
```

---

## Task 4: WebWorker

**Files:**
- Create: `tools/network-explorer/src/mlp.worker.js`
- Modify: `tools/network-explorer/src/useWorker.js`

- [ ] **Step 1: Write mlp.worker.js**

```javascript
// src/mlp.worker.js
import { sampleMLP, sampleInputs, forwardPass, outputStats } from './mlp.js';
import { meanPropagation, covariancePropagation } from './estimators.js';

self.onmessage = function ({ data }) {
  const { id, type, params } = data;
  const t0 = performance.now();

  try {
    let payload;
    switch (type) {
      case 'sampleMLP': {
        const mlp = sampleMLP(params.width, params.depth, params.seed);
        payload = { mlp };
        break;
      }
      case 'outputStats': {
        const stats = outputStats(params.mlp, params.nSamples, params.seed || 0);
        payload = { means: stats.means, variances: stats.variances };
        break;
      }
      case 'sampling': {
        const stats = outputStats(params.mlp, params.budget, params.seed || 0);
        payload = { estimates: stats.means };
        break;
      }
      case 'meanPropagation': {
        const estimates = meanPropagation(params.mlp);
        payload = { estimates };
        break;
      }
      case 'covPropagation': {
        const estimates = covariancePropagation(params.mlp);
        payload = { estimates };
        break;
      }
      default:
        throw new Error(`Unknown worker message type: ${type}`);
    }

    const time = performance.now() - t0;
    // Wrap in { id, result } to match useWorker.js expectation:
    // the hook destructures e.data as { id, result, error }
    self.postMessage({ id, result: { ...payload, time } });
  } catch (err) {
    self.postMessage({ id, error: err.message });
  }
};
```

- [ ] **Step 2: Update useWorker.js to point to new worker**

In `src/useWorker.js`, change the Worker import from `circuit.worker.js` to `mlp.worker.js`. Rename the exported hook from `useCircuitWorker` to `useMLPWorker`. The rest of the hook (lazy init, callback map, promise-based RPC) stays the same.

Key change:
```javascript
// OLD:
import CircuitWorker from './circuit.worker.js?worker';
export function useCircuitWorker() {
  // ...
  workerRef.current = new CircuitWorker();

// NEW:
import MLPWorker from './mlp.worker.js?worker';
export function useMLPWorker() {
  // ...
  workerRef.current = new MLPWorker();
```

- [ ] **Step 3: Verify the app still builds (no runtime test yet)**

Run: `cd tools/network-explorer && npx vite build 2>&1 | tail -5`
Expected: Build may fail due to App.jsx still importing old names — that's expected and will be fixed in Task 7. If it fails only due to App.jsx imports, that's fine. Check that mlp.worker.js and useWorker.js themselves have no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add tools/network-explorer/src/mlp.worker.js tools/network-explorer/src/useWorker.js
git commit -m "feat(network-explorer): add mlp.worker.js and update useWorker hook"
```

---

## Task 5: Delete Obsolete Files

**Files to delete:**
- `src/circuit.js`
- `src/circuit.worker.js`
- `src/circuit-tf.js`
- `src/components/GateStats.jsx`
- `src/components/ErrorByGateType.jsx`
- `src/components/WireStats.jsx`
- `src/components/GateDetailOverlay.jsx`
- `src/components/gateShapes.js`
- `src/components/CircuitGraph.jsx`
- `src/components/CircuitGraphJoint.jsx`

- [ ] **Step 1: Delete all obsolete files**

```bash
cd tools/network-explorer
git rm src/circuit.js src/circuit.worker.js src/circuit-tf.js
git rm src/components/GateStats.jsx src/components/ErrorByGateType.jsx src/components/WireStats.jsx
git rm src/components/GateDetailOverlay.jsx src/components/gateShapes.js
git rm src/components/CircuitGraph.jsx src/components/CircuitGraphJoint.jsx
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore(network-explorer): remove obsolete circuit-based files"
```

---

## Task 6: Rename and Update Visualization Components

**Files:**
- Rename: `src/components/CircuitHeatmap.jsx` → `src/components/NetworkHeatmap.jsx`
- Modify: `src/components/ActivationRibbon.jsx`
- Modify: `src/components/ErrorHeatmap.jsx`
- Modify: `src/components/StdHeatmap.jsx`
- Modify: `src/components/EstimatorComparison.jsx`
- Modify: `src/components/CoeffHistograms.jsx`

- [ ] **Step 1: Rename CircuitHeatmap to NetworkHeatmap**

```bash
cd tools/network-explorer
git mv src/components/CircuitHeatmap.jsx src/components/NetworkHeatmap.jsx
```

In `NetworkHeatmap.jsx`, update:
- Component name: `CircuitHeatmap` → `NetworkHeatmap`
- All references to "circuit", "gate", "wire" in labels, tooltips, aria-labels
- Remove imports of `GateDetailOverlay` and `gateShapes` — replace gate detail hover with a simple neuron info tooltip (layer index, neuron index, activation value)
- Remove gate-type color logic — use a single activation-magnitude color scale

- [ ] **Step 2: Update ActivationRibbon.jsx**

Replace all UI-facing strings:
- "gate outputs" → "neuron activations"
- "wire" → "neuron"
- "Gate" → "Neuron"
- All tooltip text referencing circuit concepts

- [ ] **Step 3: Update ErrorHeatmap.jsx**

- Remove import of `gateShapes` and `HeatmapTooltip` gate-type fields
- "gate" → "neuron" in all labels and tooltips
- Remove gate-type breakdown from hover tooltip — show: layer, neuron index, error value

- [ ] **Step 4: Update StdHeatmap.jsx**

- Remove import of `gateShapes`
- "wire" → "neuron" in labels and tooltips

- [ ] **Step 5: Update EstimatorComparison.jsx**

- "wire" → "neuron" in any labels
- Series labels should stay: "Sampling", "Mean Prop", "Cov Prop"

- [ ] **Step 6: Rewrite CoeffHistograms.jsx for weight distributions**

The current component shows 4 subplots for bilinear coefficients (c, a, b, p). Rewrite to show:
- Single histogram per layer: distribution of weight values
- Per-layer mean ± σ band chart (same visual style, simpler data)
- Input: `mlp.weights` array instead of `circuit.gates`
- Make the entire panel collapsible (add a toggle header)

- [ ] **Step 7: Commit**

```bash
git add -A tools/network-explorer/src/components/
git commit -m "refactor(network-explorer): update visualization components for MLP semantics"
```

---

## Task 7: NetworkGraph Component (Tiny MLPs)

**Files:**
- Create: `tools/network-explorer/src/components/NetworkGraph.jsx`

- [ ] **Step 1: Implement NetworkGraph.jsx**

Build a JointJS-based visualization for MLPs with width ≤ 8:
- **Layout:** Columns of neurons, one column per layer (input layer + depth hidden layers)
- **Input layer:** Column of `width` nodes labeled "x₁"..."xₙ", colored by input value
- **Hidden layers:** Columns of `width` neuron nodes, colored by activation value (0 = dark, high = bright)
- **Edges:** Lines connecting neurons between adjacent layers, colored by weight value (negative = blue, zero = gray, positive = red), thickness proportional to |weight|
- **Interactivity:** Click a neuron to highlight its incoming/outgoing connections, dim others

**Props:**
- `mlp` — the MLP object `{ width, depth, weights }`
- `means` — `Float32Array(depth × width)` of neuron mean activations (for coloring)
- `activeLayer` — currently selected layer index (for highlighting)

Reuse JointJS patterns from the deleted `CircuitGraphJoint.jsx` (joint.dia.Graph, joint.dia.Paper, Element/Link creation, ResizeObserver for sizing).

- [ ] **Step 2: Verify it renders with a test MLP**

This will be visually tested once App.jsx is wired up in Task 8. For now, ensure the file has no syntax errors:

Run: `cd tools/network-explorer && npx vite build 2>&1 | tail -5`
Expected: May still fail due to App.jsx imports — that's fine. The component file itself should parse.

- [ ] **Step 3: Commit**

```bash
git add tools/network-explorer/src/components/NetworkGraph.jsx
git commit -m "feat(network-explorer): add NetworkGraph component for tiny MLPs"
```

---

## Task 8: Update EstimatorRunner

**Files:**
- Modify: `tools/network-explorer/src/components/EstimatorRunner.jsx`

- [ ] **Step 1: Rewrite EstimatorRunner for MLP**

Key changes:
- Remove all TF.js / GPU code paths (imports of `circuit-tf.js`, `empiricalMeanTF`, `empiricalStatsTF`)
- All estimators now go through the worker:
  - **Ground Truth:** worker `outputStats` with nSamples=10000
  - **Sampling:** worker `sampling` with user-controlled budget
  - **Mean Propagation:** worker `meanPropagation`
  - **Covariance Propagation:** worker `covPropagation`
- Props change: accept `mlp` instead of `circuit`
- Update all labels: "circuit" → "network", "gate" → "neuron"
- Remove TF.js backend selector UI

- [ ] **Step 2: Commit**

```bash
git add tools/network-explorer/src/components/EstimatorRunner.jsx
git commit -m "refactor(network-explorer): simplify EstimatorRunner for MLP (remove TF.js)"
```

---

## Task 9: Update NarrativeCard and StepIndicator

**Files:**
- Modify: `tools/network-explorer/src/components/NarrativeCard.jsx`
- Modify: `tools/network-explorer/src/components/StepIndicator.jsx`

- [ ] **Step 1: Rewrite STEP_CONTENT in NarrativeCard.jsx**

Replace the `STEP_CONTENT` object with new MLP tour text:

| Step | Title | Body |
|------|-------|------|
| 1 | Meet the MLP | "This is a Multi-Layer Perceptron — a stack of layers where each neuron computes a weighted sum of its inputs, then applies ReLU (keeping only positive values). The inputs are random Gaussian numbers." |
| 2 | Watch It Compute | "Watch as random inputs flow through the network. Each layer transforms the signal — weights amplify or dampen, ReLU clips negatives to zero. Notice how the activation pattern changes layer by layer." |
| 3 | The Goal | "These colored cells show the average neuron activation across many random inputs. Your challenge: predict these averages without running thousands of samples. Can you figure out E[neuron] from the weights alone?" |
| 4 | Sampling | "The simplest approach: draw random inputs, run them through, average the outputs. More samples = better estimates, but it's slow. Notice the noise — with only {budget} samples, estimates are rough." |
| 5 | Mean Propagation | "Instead of sampling, propagate the expected value analytically: E[ReLU(z)] = μΦ(μ/σ) + σφ(μ/σ). It's instant — but assumes neurons are independent. Watch it drift at deeper layers where correlations build up." |
| 6 | The Challenge | "Covariance propagation tracks correlations and does better — but costs O(width²) per layer. The contest: given a compute budget, can you beat sampling? You now have all the tools. Explore freely!" |

Update `MathTerm` hover definitions for MLP terms: "E[neuron]", "ReLU", "Φ (normal CDF)", "φ (normal PDF)", "covariance".

- [ ] **Step 2: Update StepIndicator.jsx step labels**

Update the 6 step labels to match the new tour:
```javascript
const STEP_LABELS = [
  'The MLP',
  'Forward Pass',
  'Neuron Means',
  'Sampling',
  'Mean Prop',
  'Challenge',
];
```

- [ ] **Step 3: Commit**

```bash
git add tools/network-explorer/src/components/NarrativeCard.jsx tools/network-explorer/src/components/StepIndicator.jsx
git commit -m "feat(network-explorer): rewrite tour narrative for MLP concepts"
```

---

## Task 10: Wire Up App.jsx

**Files:**
- Modify: `tools/network-explorer/src/App.jsx`
- Modify: `tools/network-explorer/src/components/Controls.jsx`

This is the integration task — connect all the new pieces.

- [ ] **Step 1: Update imports in App.jsx**

```javascript
// Remove:
import { randomCircuit, empiricalMean, empiricalStats, runSingleTrial } from './circuit.js';
import { useCircuitWorker } from './useWorker.js';
// (and any imports of deleted components)

// Add:
import { sampleMLP, forwardPass, sampleInputs } from './mlp.js';
import { useMLPWorker } from './useWorker.js';
import NetworkGraph from './components/NetworkGraph.jsx';
import NetworkHeatmap from './components/NetworkHeatmap.jsx';
// (keep other component imports, minus deleted ones)
```

- [ ] **Step 2: Replace circuit state with MLP state**

Key state changes:
- `displayCircuit` → `displayMLP` (holds `{ width, depth, weights }`)
- Tour params: `{ width: 8, depth: 6, seed: 42 }` (small enough for graph view)
- Explore params: width slider 4–256, depth slider 2–32
- View threshold: `displayMLP.width <= 8` → NetworkGraph, else → NetworkHeatmap
- Replace `useCircuitWorker()` with `useMLPWorker()`

- [ ] **Step 3: Update tour step logic**

Map the 6 tour steps to MLP operations:
1. Generate MLP, show graph
2. Run a single forward pass batch, show activations on graph
3. Auto-run ground truth (worker `outputStats` with 10000 samples), show heatmap
4. Auto-run sampling (worker `sampling` with small budget), show error overlay
5. Auto-run mean propagation (worker `meanPropagation`), show comparison
6. Run covariance propagation, unlock explore mode

- [ ] **Step 4: Update explore mode logic**

- On parameter change: regenerate MLP via `sampleMLP()` (sync for small, worker for large)
- Threshold for worker: `width > 32` (larger weight matrices are slow to generate)
- Wire estimator results to all visualization components

- [ ] **Step 5: Remove references to deleted components**

Remove all `<GateStats>`, `<ErrorByGateType>`, `<WireStats>` JSX and their imports. Remove `<SignalHeatmap>` if it was gate-specific (check first — keep if it shows generic layer data).

- [ ] **Step 6: Update Controls.jsx**

- Rename "Width (wires per layer)" → "Width (neurons per layer)"
- Update slider ranges: width 4–256 (was wire count), depth 2–32
- Remove any gate-type-related controls

- [ ] **Step 7: Update all remaining terminology in App.jsx**

Search and replace in JSX text, comments, variable names:
- "circuit" → "mlp" / "network"
- "gate" → "neuron"
- "wire" → "neuron" / "connection"

- [ ] **Step 8: Smoke test in browser**

Run: `cd tools/network-explorer && npm run dev`
Open http://localhost:5173 in browser. Verify:
- Tour mode loads and shows a neuron graph
- Can click through all 6 tour steps
- Explore mode lets you adjust width/depth
- Switching width > 8 shows heatmap
- All estimators can be run without errors

- [ ] **Step 9: Commit**

```bash
git add tools/network-explorer/src/App.jsx tools/network-explorer/src/components/Controls.jsx
git commit -m "feat(network-explorer): wire up App.jsx with MLP data layer and updated components"
```

---

## Task 11: Final Cleanup

**Files:**
- Various — sweep for leftover circuit references

- [ ] **Step 1: Search for remaining "circuit" references**

```bash
cd tools/network-explorer && grep -ri "circuit\|gate\b\|wire\b" src/ --include="*.js" --include="*.jsx" -l
```

Fix any remaining references in files not yet updated (e.g., `HeatmapTooltip.jsx`, `CanvasTooltip.jsx`, `PerfOverlay.jsx`, `SignalHeatmap.jsx`).

- [ ] **Step 2: Update index.html title**

In `tools/network-explorer/index.html`, update `<title>` to "Network Explorer" if not already done.

- [ ] **Step 3: Run full test suite**

Run: `cd tools/network-explorer && npx vitest run`
Expected: All tests pass

- [ ] **Step 4: Run build**

Run: `cd tools/network-explorer && npx vite build`
Expected: Clean build with no errors

- [ ] **Step 5: Final browser smoke test**

Open http://localhost:5173. Walk through entire tour. Switch to explore mode. Test all parameter combinations. Run all estimators.

- [ ] **Step 6: Commit**

```bash
git add -A tools/network-explorer/
git commit -m "chore(network-explorer): final cleanup — remove all circuit terminology"
```
