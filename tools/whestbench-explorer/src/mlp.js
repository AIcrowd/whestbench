/**
 * mlp.js — MLP data layer for WhestBench Explorer
 *
 * Exports:
 *   sampleMLP(width, depth, seed)   — sample a random He-initialized MLP
 *   sampleInputs(n, width, seed)    — sample Gaussian N(0,1) input matrix
 *   forwardPass(mlp, inputs)        — forward pass returning per-layer activations
 *   outputStats(mlp, nSamples, seed) — mean and variance per neuron per layer
 *
 * Forward pass convention: x @ W per layer (row-vector, matching Python simulation.py)
 */

import { makeRng, boxMuller } from './math-utils.js';

// ── sampleMLP ─────────────────────────────────────────────────────────────

/**
 * Sample a random MLP with He-initialized weights.
 *
 * @param {number} width  — number of neurons per layer
 * @param {number} depth  — number of layers (= number of weight matrices)
 * @param {number} seed   — integer PRNG seed
 * @returns {{ width: number, depth: number, weights: Float32Array[] }}
 *   weights: array of `depth` Float32Arrays, each of length width×width (row-major).
 *   He scale: std = sqrt(2 / width).
 */
export function sampleMLP(width, depth, seed) {
  const rng = makeRng(seed);
  const scale = Math.sqrt(2 / width);
  const weights = [];

  for (let l = 0; l < depth; l++) {
    const w = new Float32Array(width * width);
    let i = 0;
    // Each call to boxMuller yields 2 standard-normal samples
    while (i < w.length) {
      const [g1, g2] = boxMuller(rng);
      w[i] = g1 * scale;
      if (i + 1 < w.length) {
        w[i + 1] = g2 * scale;
      }
      i += 2;
    }
    weights.push(w);
  }

  return { width, depth, weights };
}

// ── sampleInputs ──────────────────────────────────────────────────────────

/**
 * Sample a random Gaussian N(0,1) input matrix.
 *
 * @param {number} n      — number of input rows (samples)
 * @param {number} width  — number of input dimensions
 * @param {number} seed   — integer PRNG seed
 * @returns {Float32Array} shape (n × width), row-major
 */
export function sampleInputs(n, width, seed) {
  const rng = makeRng(seed);
  const total = n * width;
  const out = new Float32Array(total);
  let i = 0;
  while (i < total) {
    const [g1, g2] = boxMuller(rng);
    out[i] = g1;
    if (i + 1 < total) {
      out[i + 1] = g2;
    }
    i += 2;
  }
  return out;
}

// ── forwardPass ───────────────────────────────────────────────────────────

/**
 * Run a forward pass through the MLP, returning post-ReLU activations
 * after each layer.
 *
 * Convention: x_next = ReLU(x @ W)   (row-vector, matching Python simulation.py)
 *
 * @param {{ width: number, depth: number, weights: Float32Array[] }} mlp
 * @param {Float32Array} inputs — shape (n, width), row-major
 * @returns {Float32Array[]} array of `depth` Float32Arrays, each shape (n, width)
 */
export function forwardPass(mlp, inputs) {
  const { width, weights } = mlp;
  const n = inputs.length / width;
  const layerOutputs = [];

  let x = inputs;

  for (const w of weights) {
    // Compute z = x @ W, then apply ReLU
    // x: (n, width), W: (width, width) → z: (n, width)
    // z[row, col] = sum_k x[row, k] * W[k, col]
    const z = new Float32Array(n * width);
    for (let row = 0; row < n; row++) {
      const rowOffset = row * width;
      for (let col = 0; col < width; col++) {
        let sum = 0;
        for (let k = 0; k < width; k++) {
          sum += x[rowOffset + k] * w[k * width + col];
        }
        // ReLU in-place as we write
        z[rowOffset + col] = sum > 0 ? sum : 0;
      }
    }
    layerOutputs.push(z);
    x = z;
  }

  return layerOutputs;
}

// ── outputStats ───────────────────────────────────────────────────────────

/**
 * Compute per-layer, per-neuron means and variances by chunked sampling.
 *
 * Chunked to bound memory to O(chunk_size × width). Uses Float64Array
 * accumulators for numerical stability.
 *
 * @param {{ width: number, depth: number, weights: Float32Array[] }} mlp
 * @param {number} nSamples   — total number of Gaussian input samples
 * @param {number} [seed=0]   — PRNG seed for reproducibility
 * @returns {{ means: Float32Array, variances: Float32Array }}
 *   Both are shape (depth × width), layer-major, row-major within a layer:
 *   index = layer * width + neuron
 */
export function outputStats(mlp, nSamples, seed = 0, onProgress = null) {
  const { width, depth } = mlp;
  const chunkSize = Math.min(nSamples, 512);

  // Float64 accumulators for numerical stability (Welford-style two-pass via
  // sum + sum-of-squares approach; exact two-pass is fine at chunk granularity)
  const sumAcc = new Float64Array(depth * width);
  const sumSqAcc = new Float64Array(depth * width);

  let totalSamples = 0;
  let samplesSoFar = 0;
  let chunkSeed = seed;

  while (samplesSoFar < nSamples) {
    const thisChunk = Math.min(chunkSize, nSamples - samplesSoFar);

    // Sample inputs for this chunk
    const inputs = sampleInputs(thisChunk, width, chunkSeed);
    // Advance seed so next chunk differs
    chunkSeed = chunkSeed + 1;

    // Forward pass
    const layerOutputs = forwardPass(mlp, inputs);

    // Accumulate sums and sum-of-squares per layer per neuron
    for (let l = 0; l < depth; l++) {
      const out = layerOutputs[l]; // Float32Array (thisChunk × width)
      const base = l * width;
      for (let row = 0; row < thisChunk; row++) {
        const rowOffset = row * width;
        for (let neuron = 0; neuron < width; neuron++) {
          const v = out[rowOffset + neuron];
          sumAcc[base + neuron] += v;
          sumSqAcc[base + neuron] += v * v;
        }
      }
    }

    samplesSoFar += thisChunk;
    totalSamples += thisChunk;
    if (onProgress) onProgress(samplesSoFar / nSamples);
  }

  // Compute final means and variances from accumulators
  const means = new Float32Array(depth * width);
  const variances = new Float32Array(depth * width);

  for (let i = 0; i < depth * width; i++) {
    const mean = sumAcc[i] / totalSamples;
    const variance = sumSqAcc[i] / totalSamples - mean * mean;
    means[i] = mean;
    variances[i] = Math.max(0, variance); // clamp to avoid tiny negatives from float precision
  }

  return { means, variances };
}
