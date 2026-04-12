import { describe, it, expect } from 'vitest';
import { sampleMLP, sampleInputs, forwardPass, outputStats } from '../mlp.js';

// ── sampleMLP ──────────────────────────────────────────────────────────────

describe('sampleMLP', () => {
  it('returns correct structure', () => {
    const mlp = sampleMLP(8, 3, 42);
    expect(mlp.width).toBe(8);
    expect(mlp.depth).toBe(3);
    expect(Array.isArray(mlp.weights)).toBe(true);
    expect(mlp.weights).toHaveLength(3);
  });

  it('each weight matrix is a Float32Array of length width × width', () => {
    const mlp = sampleMLP(16, 2, 1);
    for (const w of mlp.weights) {
      expect(w).toBeInstanceOf(Float32Array);
      expect(w.length).toBe(16 * 16);
    }
  });

  it('is deterministic with the same seed', () => {
    const mlp1 = sampleMLP(8, 2, 99);
    const mlp2 = sampleMLP(8, 2, 99);
    for (let l = 0; l < mlp1.depth; l++) {
      expect(Array.from(mlp1.weights[l])).toEqual(Array.from(mlp2.weights[l]));
    }
  });

  it('produces different weights with a different seed', () => {
    const mlp1 = sampleMLP(8, 2, 42);
    const mlp2 = sampleMLP(8, 2, 43);
    // Very unlikely that first weight arrays are equal
    expect(Array.from(mlp1.weights[0])).not.toEqual(Array.from(mlp2.weights[0]));
  });

  it('He initialization: variance ≈ 2/width (width=256)', () => {
    const width = 256;
    const mlp = sampleMLP(width, 1, 7);
    const w = mlp.weights[0];
    // Compute sample variance of all weights in this layer
    let sum = 0;
    let sumSq = 0;
    const n = w.length;
    for (let i = 0; i < n; i++) {
      sum += w[i];
      sumSq += w[i] * w[i];
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    const expected = 2 / width;
    // Allow ±30% relative tolerance for a large sample
    expect(variance).toBeGreaterThan(expected * 0.7);
    expect(variance).toBeLessThan(expected * 1.3);
  });
});

// ── sampleInputs ───────────────────────────────────────────────────────────

describe('sampleInputs', () => {
  it('returns a Float32Array of length n × width', () => {
    const inputs = sampleInputs(50, 8, 0);
    expect(inputs).toBeInstanceOf(Float32Array);
    expect(inputs.length).toBe(50 * 8);
  });

  it('is deterministic with the same seed', () => {
    const a = sampleInputs(100, 4, 42);
    const b = sampleInputs(100, 4, 42);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('produces different values with a different seed', () => {
    const a = sampleInputs(100, 4, 0);
    const b = sampleInputs(100, 4, 1);
    expect(Array.from(a)).not.toEqual(Array.from(b));
  });

  it('approximately standard normal over 10k samples (mean ≈ 0, variance ≈ 1)', () => {
    const n = 10000;
    const width = 1;
    const inputs = sampleInputs(n, width, 42);
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < inputs.length; i++) {
      sum += inputs[i];
      sumSq += inputs[i] * inputs[i];
    }
    const mean = sum / inputs.length;
    const variance = sumSq / inputs.length - mean * mean;
    expect(mean).toBeCloseTo(0, 1);    // within 0.1
    expect(variance).toBeCloseTo(1, 1); // within 0.1
  });
});

// ── forwardPass ────────────────────────────────────────────────────────────

describe('forwardPass', () => {
  it('returns an array of depth Float32Arrays', () => {
    const mlp = sampleMLP(8, 3, 1);
    const inputs = sampleInputs(10, 8, 2);
    const outputs = forwardPass(mlp, inputs);
    expect(Array.isArray(outputs)).toBe(true);
    expect(outputs).toHaveLength(3);
    for (const layer of outputs) {
      expect(layer).toBeInstanceOf(Float32Array);
    }
  });

  it('each layer output has length n × width', () => {
    const width = 8;
    const n = 10;
    const mlp = sampleMLP(width, 2, 3);
    const inputs = sampleInputs(n, width, 5);
    const outputs = forwardPass(mlp, inputs);
    for (const layer of outputs) {
      expect(layer.length).toBe(n * width);
    }
  });

  it('all outputs are ≥ 0 (ReLU)', () => {
    const mlp = sampleMLP(16, 4, 0);
    const inputs = sampleInputs(20, 16, 1);
    const outputs = forwardPass(mlp, inputs);
    for (const layer of outputs) {
      for (let i = 0; i < layer.length; i++) {
        expect(layer[i]).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('is deterministic', () => {
    const mlp = sampleMLP(8, 2, 7);
    const inputs = sampleInputs(5, 8, 7);
    const out1 = forwardPass(mlp, inputs);
    const out2 = forwardPass(mlp, inputs);
    for (let l = 0; l < out1.length; l++) {
      expect(Array.from(out1[l])).toEqual(Array.from(out2[l]));
    }
  });

  it('matches row-vector convention: x @ W per layer', () => {
    // Construct a simple 2×2 MLP with known weights
    const width = 2;
    const depth = 1;
    // Weight matrix (row-major): W = [[1, 0], [0, 1]] (identity)
    const weights = [new Float32Array([1, 0, 0, 1])];
    const mlp = { width, depth, weights };
    // Single input row: x = [1, -2]
    // x @ I = [1, -2], after ReLU = [1, 0]
    const inputs = new Float32Array([1, -2]);
    const outputs = forwardPass(mlp, inputs);
    expect(outputs).toHaveLength(1);
    expect(outputs[0][0]).toBeCloseTo(1, 5);
    expect(outputs[0][1]).toBeCloseTo(0, 5);
  });

  it('matches row-vector convention with a non-trivial weight matrix', () => {
    // W = [[2, 3], [-1, 4]]  (row-major: [2, 3, -1, 4])
    // x = [1, 2]
    // x @ W = [1*2 + 2*(-1), 1*3 + 2*4] = [0, 11]
    // ReLU([0, 11]) = [0, 11]
    const width = 2;
    const weights = [new Float32Array([2, 3, -1, 4])];
    const mlp = { width, depth: 1, weights };
    const inputs = new Float32Array([1, 2]);
    const outputs = forwardPass(mlp, inputs);
    expect(outputs[0][0]).toBeCloseTo(0, 5);
    expect(outputs[0][1]).toBeCloseTo(11, 5);
  });
});

// ── outputStats ────────────────────────────────────────────────────────────

describe('outputStats', () => {
  it('returns means and variances with correct shape (depth × width)', () => {
    const depth = 3;
    const width = 8;
    const mlp = sampleMLP(width, depth, 42);
    const { means, variances } = outputStats(mlp, 200, 0);
    expect(means).toBeInstanceOf(Float32Array);
    expect(variances).toBeInstanceOf(Float32Array);
    expect(means.length).toBe(depth * width);
    expect(variances.length).toBe(depth * width);
  });

  it('all values are finite', () => {
    const mlp = sampleMLP(8, 2, 1);
    const { means, variances } = outputStats(mlp, 100, 0);
    for (let i = 0; i < means.length; i++) {
      expect(isFinite(means[i])).toBe(true);
    }
    for (let i = 0; i < variances.length; i++) {
      expect(isFinite(variances[i])).toBe(true);
    }
  });

  it('all variances are ≥ 0', () => {
    const mlp = sampleMLP(8, 2, 5);
    const { variances } = outputStats(mlp, 100, 0);
    for (let i = 0; i < variances.length; i++) {
      expect(variances[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('is deterministic with the same seed', () => {
    const mlp = sampleMLP(8, 2, 10);
    const { means: m1, variances: v1 } = outputStats(mlp, 100, 77);
    const { means: m2, variances: v2 } = outputStats(mlp, 100, 77);
    expect(Array.from(m1)).toEqual(Array.from(m2));
    expect(Array.from(v1)).toEqual(Array.from(v2));
  });

  it('means for ReLU network are ≥ 0 (post-ReLU activations have non-negative means)', () => {
    const mlp = sampleMLP(16, 3, 42);
    const { means } = outputStats(mlp, 500, 0);
    for (let i = 0; i < means.length; i++) {
      expect(means[i]).toBeGreaterThanOrEqual(0);
    }
  });
});
