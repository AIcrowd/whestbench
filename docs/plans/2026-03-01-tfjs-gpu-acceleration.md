# TF.js GPU-Accelerated Sampling — Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Replace the scalar JS `empiricalMean` with TF.js GPU-accelerated tensor operations, achieving 10-50× speedup for large circuits, plus progressive streaming UX.

**Architecture:** New `circuit-tf.js` module with `runBatchedTF` and `empiricalMeanTF` — a 1:1 port of the Python numpy code using TF.js `tf.gather` + element-wise ops. The existing `circuit.js` remains as CPU fallback. Worker dispatches to TF.js path when available.

**Tech Stack:** TensorFlow.js (`@tensorflow/tfjs`), Vite, React

---

### Task 1: Install TF.js and Verify Build

**Files:**
- Modify: `tools/circuit-explorer/package.json`

**Step 1: Install @tensorflow/tfjs**

```bash
cd tools/circuit-explorer
npm install @tensorflow/tfjs
```

**Step 2: Verify the Vite build still works**

```bash
npm run build
```

Expected: Build succeeds with no errors. TF.js may show warnings about unused exports — that's fine.

**Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "deps: add @tensorflow/tfjs for GPU-accelerated sampling"
```

---

### Task 2: Create `circuit-tf.js` — Core GPU Functions

**Files:**
- Create: `tools/circuit-explorer/src/circuit-tf.js`

**Step 1: Write `initTF()` and `runBatchedTF()`**

```javascript
/**
 * circuit-tf.js — GPU-accelerated circuit evaluation via TensorFlow.js
 *
 * Drop-in replacement for empiricalMean/runBatched from circuit.js.
 * Uses tf.gather + element-wise ops to evaluate all trials in parallel.
 */
import * as tf from '@tensorflow/tfjs';

let backendName = null;

/**
 * Initialize TF.js backend. Call once on app mount.
 * Returns the selected backend name ('webgpu' | 'webgl' | 'cpu').
 */
export async function initTF() {
  if (backendName) return backendName;

  // Try WebGPU first (3× faster than WebGL), fall back to WebGL, then CPU
  for (const backend of ['webgpu', 'webgl', 'cpu']) {
    try {
      const ok = await tf.setBackend(backend);
      if (ok) {
        await tf.ready();
        backendName = backend;
        console.log(`[circuit-tf] Using backend: ${backend}`);
        return backend;
      }
    } catch {
      // Try next backend
    }
  }
  throw new Error('No TF.js backend available');
}

/**
 * Convert a circuit layer's typed arrays into TF.js tensors.
 * Returns tensors that must be disposed after use.
 */
function layerToTensors(layer, n) {
  return {
    firstIdx: tf.tensor1d(Array.from(layer.first), 'int32'),
    secondIdx: tf.tensor1d(Array.from(layer.second), 'int32'),
    constT: tf.tensor1d(layer.const),
    firstCoeffT: tf.tensor1d(layer.firstCoeff),
    secondCoeffT: tf.tensor1d(layer.secondCoeff),
    productCoeffT: tf.tensor1d(layer.productCoeff),
  };
}

/**
 * Run circuit on batched inputs using TF.js tensors.
 * inputs: tf.Tensor2D of shape [trials, n]
 * Returns: array of tf.Tensor2D (one per layer, each [trials, n])
 *
 * Caller must dispose returned tensors.
 */
export function runBatchedTF(circuit, inputs) {
  const results = [];
  let x = inputs;

  for (const layer of circuit.gates) {
    const lt = layerToTensors(layer, circuit.n);

    const newX = tf.tidy(() => {
      const xFirst = tf.gather(x, lt.firstIdx, 1);    // [trials, n]
      const xSecond = tf.gather(x, lt.secondIdx, 1);   // [trials, n]

      return tf.add(
        tf.add(
          tf.add(lt.constT, tf.mul(lt.firstCoeffT, xFirst)),
          tf.mul(lt.secondCoeffT, xSecond)
        ),
        tf.mul(lt.productCoeffT, tf.mul(xFirst, xSecond))
      );
    });

    // Dispose layer tensors
    Object.values(lt).forEach(t => t.dispose());

    // Dispose previous x (but not the original inputs)
    if (x !== inputs) x.dispose();
    x = newX;
    results.push(tf.keep(newX));
  }

  return results;
}

/**
 * GPU-accelerated empirical mean estimation.
 * Generates random ±1 inputs, runs through circuit, returns per-layer means.
 *
 * Returns: Float32Array[] — one per layer, each of length n.
 * onProgress: optional callback(layerIdx, totalLayers) for streaming.
 */
export async function empiricalMeanTF(circuit, trials, seed, onProgress) {
  await initTF();

  // Generate random ±1 inputs: [trials, n]
  // TF.js doesn't have a seedable RNG, so we generate on CPU and transfer
  const inputData = new Float32Array(trials * circuit.n);
  // Simple seedable RNG (same xoshiro128** as circuit.js)
  let s = [seed, seed ^ 0xdeadbeef, seed ^ 0xcafebabe, seed ^ 0x12345678];
  for (let i = 0; i < 20; i++) {
    const result = (s[1] * 5) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 11) | (s[3] >>> 21);
  }
  for (let i = 0; i < inputData.length; i++) {
    const result = (s[1] * 5) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 11) | (s[3] >>> 21);
    inputData[i] = ((result >>> 0) / 0x100000000) < 0.5 ? -1.0 : 1.0;
  }

  const inputTensor = tf.tensor2d(inputData, [trials, circuit.n]);

  // Run batched through all layers
  const layerOutputs = runBatchedTF(circuit, inputTensor);
  inputTensor.dispose();

  // Compute mean per wire for each layer
  const means = [];
  for (let l = 0; l < layerOutputs.length; l++) {
    const meanTensor = tf.mean(layerOutputs[l], 0); // [n]
    const data = await meanTensor.data();             // Float32Array
    means.push(Float32Array.from(data));
    meanTensor.dispose();
    layerOutputs[l].dispose();
    if (onProgress) onProgress(l + 1, layerOutputs.length);
  }

  return means;
}

/**
 * Check if TF.js is available and initialized.
 */
export function isTFReady() {
  return backendName !== null;
}

export function getTFBackend() {
  return backendName;
}
```

**Step 2: Verify build**

```bash
cd tools/circuit-explorer && npm run build
```

Expected: Build succeeds.

**Step 3: Commit**

```bash
git add src/circuit-tf.js
git commit -m "feat: add circuit-tf.js with GPU-accelerated empiricalMeanTF"
```

---

### Task 3: Wire TF.js into the Worker

**Files:**
- Modify: `tools/circuit-explorer/src/circuit.worker.js`

**Step 1: Add TF.js import and new message handler**

Add a new case `'empiricalMeanTF'` to the worker's switch statement. This must be async because TF.js operations return Promises:

```javascript
import { empiricalMeanTF, initTF, isTFReady } from './circuit-tf';

// Add to the switch statement:
case 'empiricalMeanTF': {
  const circuit = params.circuit;
  const result = { estimates: await empiricalMeanTF(circuit, params.trials, params.seed) };
  break;
}

case 'initTF': {
  const backend = await initTF();
  result = { backend };
  break;
}
```

The `onmessage` handler must become async to support `await`:

```javascript
self.onmessage = async function (e) {
  // ... existing switch with new cases
};
```

**Step 2: Verify build**

```bash
npm run build
```

Expected: Build succeeds.

**Step 3: Commit**

```bash
git add src/circuit.worker.js
git commit -m "feat: wire TF.js empiricalMeanTF into the web worker"
```

---

### Task 4: Add TF.js Path to EstimatorRunner UI

**Files:**
- Modify: `tools/circuit-explorer/src/components/EstimatorRunner.jsx`
- Modify: `tools/circuit-explorer/src/App.jsx`

**Step 1: Initialize TF.js on app mount**

In `App.jsx`, add a useEffect that sends `initTF` to the worker on first render:

```javascript
// In App component, after worker is created:
useEffect(() => {
  if (workerRun) {
    workerRun('initTF', {}).then(result => {
      console.log('[App] TF.js backend:', result.backend);
    }).catch(err => {
      console.warn('[App] TF.js init failed, using CPU fallback:', err);
    });
  }
}, [workerRun]);
```

**Step 2: Update EstimatorRunner to use TF.js path**

Modify the `runGroundTruth` and `runSampling` callbacks to use `'empiricalMeanTF'` type instead of `'empiricalMean'`:

```javascript
const runGroundTruth = useCallback(() => {
  const gtBudget = 10000;
  runEstimator("groundTruth", "empiricalMeanTF",
    { circuit, trials: gtBudget, seed: 7777 },
    { name: "Ground Truth (10k samples)", budget: gtBudget }
  );
}, [circuit, runEstimator]);

const runSampling = useCallback(() => {
  runEstimator("sampling", "empiricalMeanTF",
    { circuit, trials: budget, seed: 1234 },
    { name: "Sampling", budget }
  );
}, [circuit, budget, runEstimator]);
```

**Step 3: Verify build and test in browser**

```bash
npm run build
```

Then open http://localhost:5179/, set a circuit to 128×128, run Ground Truth. It should complete significantly faster than before.

**Step 4: Commit**

```bash
git add src/App.jsx src/components/EstimatorRunner.jsx
git commit -m "feat: use TF.js empiricalMeanTF for ground truth and sampling"
```

---

### Task 5: Add TF.js Rows to Benchmark

**Files:**
- Modify: `tools/circuit-explorer/benchmark.html`

**Step 1: Add TF.js benchmark section**

Import `empiricalMeanTF` and `initTF` from `circuit-tf.js`. Add a new benchmark table "Estimator Computation (TF.js GPU)" and run the same circuit sizes through `empiricalMeanTF` for comparison.

**Step 2: Run benchmark**

Open http://localhost:5179/benchmark.html, click Run. Compare:
- "Estimator Computation (Web Worker)" vs "Estimator Computation (TF.js GPU)"
- Expected: TF.js rows should be 5-50× faster for sizes ≥ 64×64

**Step 3: Commit**

```bash
git add benchmark.html
git commit -m "bench: add TF.js GPU rows to benchmark for comparison"
```

---

### Task 6: Numerical Accuracy Verification

**Files:**
- Modify: `tools/circuit-explorer/benchmark.html`

**Step 1: Add accuracy comparison**

For each circuit size, run both `empiricalMean` (CPU) and `empiricalMeanTF` (GPU) with the same seed and trials, then compute max absolute difference across all wire means. Add a "Max Δ" column.

**Step 2: Verify accuracy**

Open benchmark, confirm max absolute difference is < 0.05 for all sizes. (Not exact equality because TF.js uses float32 GPU arithmetic vs JS float64.)

**Step 3: Commit**

```bash
git add benchmark.html
git commit -m "bench: add numerical accuracy comparison between CPU and TF.js paths"
```

---

### Task 7: Progressive Streaming UX (Optional Enhancement)

> This task is optional — implement only if Tasks 1-6 are successful and the speedup alone isn't sufficient for UX.

**Files:**
- Modify: `tools/circuit-explorer/src/circuit-tf.js`
- Modify: `tools/circuit-explorer/src/components/EstimatorRunner.jsx`

**Step 1: Implement batched streaming in `empiricalMeanTF`**

Split trials into batches of 500, compute running average, and call `onProgress` after each batch with the partial results.

**Step 2: Update EstimatorRunner UI**

Show a progress bar with `{completed}/{total} trials` and update the heatmap after each batch callback.

**Step 3: Test with 256×128, 10k trials**

Verify heatmap updates progressively — starts noisy, converges to final values.

**Step 4: Commit**

```bash
git add src/circuit-tf.js src/components/EstimatorRunner.jsx
git commit -m "feat: progressive streaming for empiricalMeanTF with live heatmap updates"
```
