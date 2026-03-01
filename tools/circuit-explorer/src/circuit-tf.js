/**
 * circuit-tf.js — GPU-accelerated circuit evaluation via TensorFlow.js
 *
 * Drop-in replacement for empiricalMean/runBatched from circuit.js.
 * Uses tf.gather + element-wise ops to evaluate all trials in parallel.
 *
 * The GPU path evaluates the same gate equation as the CPU path:
 *   out[i] = c[i] + a[i]*x[first[i]] + b[i]*x[second[i]] + p[i]*x[first[i]]*x[second[i]]
 *
 * But instead of looping over (trials × wires), it does a single
 * tensor gather + broadcast multiply per layer — all trials in parallel.
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

// ── Seedable PRNG (xoshiro128**, same as circuit.js) ──

function makeRng(seed = 42) {
  let s = [seed, seed ^ 0xdeadbeef, seed ^ 0xcafebabe, seed ^ 0x12345678];
  function next() {
    const result = (s[1] * 5) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 11) | (s[3] >>> 21);
    return (result >>> 0) / 0x100000000;
  }
  // warm up
  for (let i = 0; i < 20; i++) next();
  return { random: next };
}

/**
 * GPU-accelerated empirical mean estimation.
 * Generates random ±1 inputs, runs through circuit, returns per-layer means.
 *
 * Instead of storing all layer output tensors, we compute the mean inline
 * per layer and immediately pull results to CPU. This avoids complex tensor
 * lifecycle issues and keeps GPU memory minimal.
 *
 * Returns: Float32Array[] — one per layer, each of length n.
 * Compatible with the CPU empiricalMean function's return type.
 */
export async function empiricalMeanTF(circuit, trials, seed = 99, onProgress = null) {
  await initTF();

  // Generate random ±1 inputs on CPU with seedable RNG, then transfer to GPU
  const rng = makeRng(seed);
  const inputData = new Float32Array(trials * circuit.n);
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = rng.random() < 0.5 ? -1.0 : 1.0;
  }

  let x = tf.tensor2d(inputData, [trials, circuit.n]);
  const means = [];
  const totalLayers = circuit.gates.length;

  for (let li = 0; li < totalLayers; li++) {
    const layer = circuit.gates[li];

    // Extract layer data into plain JS arrays to avoid typed array issues
    const n = circuit.n;
    const firstArr = new Array(n);
    const secondArr = new Array(n);
    const constArr = new Array(n);
    const aArr = new Array(n);
    const bArr = new Array(n);
    const pArr = new Array(n);

    for (let i = 0; i < n; i++) {
      firstArr[i] = layer.first[i];
      secondArr[i] = layer.second[i];
      constArr[i] = layer['const'][i];
      aArr[i] = layer.firstCoeff[i];
      bArr[i] = layer.secondCoeff[i];
      pArr[i] = layer.productCoeff[i];
    }

    // Compute newX = c + a*x[first] + b*x[second] + p*x[first]*x[second]
    // All inside tf.tidy except the result, which we return from tidy
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

    // Compute mean of this layer and pull to CPU
    const meanT = tf.mean(newX, 0);
    const data = await meanT.data();
    means.push(Float32Array.from(data));
    meanT.dispose();

    // Dispose old x (but tidy should have cleaned intermediates)
    x.dispose();
    x = newX;

    // Report progress (negligible cost — one function call per layer)
    if (onProgress) {
      onProgress((li + 1) / totalLayers);
    }
  }

  // Dispose final layer output
  x.dispose();

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

