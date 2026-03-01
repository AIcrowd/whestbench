# Circuit Explorer: TF.js GPU-Accelerated Sampling

## Problem

The `empiricalMean` estimator is 100× slower than all other operations in the Circuit Explorer. At 512×256 with 10k trials, it takes **6.88 seconds** — while all rendering operations combined take <22ms. The root cause is a scalar triple-nested JS loop evaluating `trials × layers × wires` gates one at a time, versus the Python reference which uses numpy vectorized `float16` array operations on 2D matrices.

## Approach

Port `runBatched` and `empiricalMean` to **TensorFlow.js tensor operations** that run on GPU via WebGL/WebGPU backends. The numpy-to-TF.js mapping is nearly 1:1:

```python
# Python (numpy) — vectorized across all trials + wires
x = layer.const + layer.first_coeff * x[:, layer.first] \
  + layer.second_coeff * x[:, layer.second] \
  + layer.product_coeff * x[:, layer.first] * x[:, layer.second]
```

```javascript
// TF.js — same structure, runs on GPU
const xFirst  = tf.gather(x, firstIdx, 1);   // [trials, n]
const xSecond = tf.gather(x, secondIdx, 1);   // [trials, n]
x = tf.add(tf.add(tf.add(
  constTensor,
  tf.mul(firstCoeff, xFirst)),
  tf.mul(secondCoeff, xSecond)),
  tf.mul(productCoeff, tf.mul(xFirst, xSecond))
);
```

Expected speedup: **10-50× for large circuits** (GPU parallelizes across all trials and wires simultaneously).

## Architecture

### New module: `src/circuit-tf.js`

A drop-in replacement for `empiricalMean` and `runBatched` that uses TF.js tensors instead of scalar loops. The existing `circuit.js` stays untouched as a fallback.

```
circuit.js          → scalar/CPU (existing, unchanged)
circuit-tf.js       → TF.js tensor/GPU (new)
circuit.worker.js   → dispatches to circuit-tf.js when available
```

### TF.js backend selection

```javascript
await tf.setBackend('webgpu');   // preferred: modern GPUs
// fallback:
await tf.setBackend('webgl');    // universal: all browsers
// fallback:
await tf.setBackend('cpu');      // same as current, but with TF.js overhead
```

### Key functions in `circuit-tf.js`

1. **`runBatchedTF(circuit, inputs)`** — converts circuit layers to tensors once, then evaluates all trials in parallel per layer using `tf.gather` + element-wise ops.

2. **`empiricalMeanTF(circuit, trials, seed, onProgress?)`** — generates random ±1 inputs as a `[trials, n]` tensor, runs through `runBatchedTF`, reduces with `tf.mean(axis=0)` per layer. Optionally calls `onProgress(layerIdx, totalLayers)` for streaming.

3. **`initTF()`** — initializes TF.js backend (async, done once on app mount). Returns the selected backend name for display.

### Tensor memory management

TF.js requires explicit disposal of tensors. Use `tf.tidy()` to wrap computation blocks:

```javascript
const result = tf.tidy(() => {
  const x = tf.randomUniform([trials, n], -1, 1).sign();
  // ... layer-by-layer computation
  return means; // only returned tensors survive tidy()
});
```

For intermediate layer results needed for streaming, use `tf.keep()` to prevent disposal, then manually `dispose()` after use.

## Progressive UX

### Streaming results

Rather than blocking for 10k trials, process in batches (e.g., 500 trials each) and update the heatmap after each batch with the running average:

```
Batch 1:   500 trials → show preliminary heatmap (noisy)
Batch 2:  1000 trials → update heatmap (smoother)
...
Batch 20: 10000 trials → final heatmap (converged)
```

### UI changes

- Progress bar shows `completed / total` trials during estimation
- Heatmap updates live as batches complete (progressive refinement)
- "Cancel" button to stop early if estimate looks good enough

### Adaptive budget

For circuits > 128×128, auto-suggest a lower trial count with a warning:
> "Large circuit detected. Using 1,000 trials (est. ~200ms). Increase budget for higher accuracy."

## Fallback strategy

If TF.js fails to initialize (old browser, no GPU), fall back to the existing scalar `circuit.js` implementation. The user sees a small badge: "CPU mode — larger circuits may be slow."

## Files changed

| File | Change |
|------|--------|
| `package.json` | Add `@tensorflow/tfjs` dependency |
| `src/circuit-tf.js` | **[NEW]** TF.js tensor-based `runBatchedTF`, `empiricalMeanTF`, `initTF` |
| `src/circuit.worker.js` | Import `circuit-tf.js`, use TF.js path when available |
| `src/App.jsx` | Call `initTF()` on mount, pass backend info to UI |
| `src/components/EstimatorRunner.jsx` | Progressive UX: progress bar, batch streaming, cancel button |
| `benchmark.html` | Add TF.js benchmark rows for comparison |

## Verification

### Automated: benchmark comparison

Run `benchmark.html` with and without TF.js to compare:
- Same circuit, same seed → results should match within float tolerance
- TF.js path should be ≥5× faster for circuits ≥ 128×128

### Manual: visual correctness

1. Open Circuit Explorer at http://localhost:5179/
2. Set circuit to 256×128
3. Run Ground Truth (10k) → heatmap should appear progressively
4. Compare heatmap colors with the existing (slow) CPU path — should be identical
5. Run Mean Propagation → should still be instant (unchanged path)

### Numerical accuracy

The TF.js results should match the scalar JS results within ±0.01 for the same seed. This validates that the GPU computation produces the same gate evaluations as the CPU path.
