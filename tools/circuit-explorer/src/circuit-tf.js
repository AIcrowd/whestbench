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
 * Run circuit on batched inputs using TF.js tensors.
 * inputs: tf.Tensor2D of shape [trials, n]
 * Returns: array of tf.Tensor2D (one per layer, each [trials, n])
 *
 * Caller must dispose returned tensors.
 */
export function runBatchedTF(circuit, inputTensor) {
  const results = [];
  let x = inputTensor;

  for (const layer of circuit.gates) {
    // Convert layer arrays to tensors (1D, will broadcast across trials)
    const firstIdx = tf.tensor1d(Array.from(layer.first), 'int32');
    const secondIdx = tf.tensor1d(Array.from(layer.second), 'int32');
    const constT = tf.tensor1d(layer.const);
    const firstCoeffT = tf.tensor1d(layer.firstCoeff);
    const secondCoeffT = tf.tensor1d(layer.secondCoeff);
    const productCoeffT = tf.tensor1d(layer.productCoeff);

    const newX = tf.tidy(() => {
      const xFirst = tf.gather(x, firstIdx, 1);    // [trials, n]
      const xSecond = tf.gather(x, secondIdx, 1);   // [trials, n]

      // out = const + firstCoeff * xFirst + secondCoeff * xSecond
      //       + productCoeff * xFirst * xSecond
      return tf.add(
        tf.add(
          tf.add(constT, tf.mul(firstCoeffT, xFirst)),
          tf.mul(secondCoeffT, xSecond)
        ),
        tf.mul(productCoeffT, tf.mul(xFirst, xSecond))
      );
    });

    // Dispose layer tensors
    firstIdx.dispose();
    secondIdx.dispose();
    constT.dispose();
    firstCoeffT.dispose();
    secondCoeffT.dispose();
    productCoeffT.dispose();

    // Dispose previous x (but not the original inputs)
    if (x !== inputTensor) x.dispose();
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
 * Compatible with the CPU empiricalMean function's return type.
 */
export async function empiricalMeanTF(circuit, trials, seed = 99) {
  await initTF();

  // Generate random ±1 inputs on CPU with seedable RNG, then transfer to GPU
  const rng = makeRng(seed);
  const inputData = new Float32Array(trials * circuit.n);
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = rng.random() < 0.5 ? -1.0 : 1.0;
  }

  const inputTensor = tf.tensor2d(inputData, [trials, circuit.n]);

  // Run batched through all layers on GPU
  const layerOutputs = runBatchedTF(circuit, inputTensor);
  inputTensor.dispose();

  // Compute mean per wire for each layer, then pull back to CPU
  const means = [];
  for (let l = 0; l < layerOutputs.length; l++) {
    const meanTensor = tf.mean(layerOutputs[l], 0); // [n]
    const data = await meanTensor.data();             // Float32Array
    means.push(Float32Array.from(data));
    meanTensor.dispose();
    layerOutputs[l].dispose();
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
