import { describe, it, expect } from 'vitest';
import { meanPropagation, covariancePropagation } from '../estimators.js';
import { sampleMLP, outputStats } from '../mlp.js';

// ── helpers ──────────────────────────────────────────────────────────────────

/** Compute mean squared error between two Float32Arrays. */
function mse(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum / a.length;
}

// ── meanPropagation ──────────────────────────────────────────────────────────

describe('meanPropagation', () => {
  it('returns a Float32Array', () => {
    const mlp = sampleMLP(8, 2, 42);
    const result = meanPropagation(mlp);
    expect(result).toBeInstanceOf(Float32Array);
  });

  it('has shape depth × width', () => {
    const width = 16;
    const depth = 3;
    const mlp = sampleMLP(width, depth, 1);
    const result = meanPropagation(mlp);
    expect(result.length).toBe(depth * width);
  });

  it('all values ≥ 0 (post-ReLU means must be non-negative)', () => {
    const mlp = sampleMLP(16, 4, 7);
    const result = meanPropagation(mlp);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('all values are finite', () => {
    const mlp = sampleMLP(16, 4, 99);
    const result = meanPropagation(mlp);
    for (let i = 0; i < result.length; i++) {
      expect(isFinite(result[i])).toBe(true);
    }
  });

  it('is deterministic (same mlp → same result)', () => {
    const mlp = sampleMLP(8, 3, 55);
    const r1 = meanPropagation(mlp);
    const r2 = meanPropagation(mlp);
    expect(Array.from(r1)).toEqual(Array.from(r2));
  });

  it('works for depth=1', () => {
    const mlp = sampleMLP(8, 1, 0);
    const result = meanPropagation(mlp);
    expect(result.length).toBe(8);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('works for depth=8 (deep network)', () => {
    const mlp = sampleMLP(16, 8, 3);
    const result = meanPropagation(mlp);
    expect(result.length).toBe(16 * 8);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('known single-layer case: identity weights with N(0,1) inputs give E[ReLU(z)] ≈ 1/sqrt(2pi) ≈ 0.3989', () => {
    // With identity weights: pre-activation is N(0,1), so E[ReLU(z)] = phi(0) = 1/sqrt(2pi)
    const width = 4;
    const weights = [new Float32Array(width * width)];
    // Identity: W[i,i] = 1
    for (let i = 0; i < width; i++) weights[0][i * width + i] = 1.0;
    const mlp = { width, depth: 1, weights };
    const result = meanPropagation(mlp);
    const expected = 1.0 / Math.sqrt(2 * Math.PI); // ~0.3989
    for (let j = 0; j < width; j++) {
      expect(result[j]).toBeCloseTo(expected, 3);
    }
  });

  it('close to ground truth for shallow network (width=16, depth=2, MSE < 0.01)', () => {
    const width = 16;
    const depth = 2;
    const mlp = sampleMLP(width, depth, 42);

    // Ground truth: 50k Monte Carlo samples
    const { means: groundTruth } = outputStats(mlp, 50000, 0);
    const predicted = meanPropagation(mlp);

    const error = mse(predicted, groundTruth);
    expect(error).toBeLessThan(0.01);
  });
});

// ── covariancePropagation ─────────────────────────────────────────────────────

describe('covariancePropagation', () => {
  it('returns a Float32Array', () => {
    const mlp = sampleMLP(8, 2, 42);
    const result = covariancePropagation(mlp);
    expect(result).toBeInstanceOf(Float32Array);
  });

  it('has shape depth × width', () => {
    const width = 16;
    const depth = 3;
    const mlp = sampleMLP(width, depth, 2);
    const result = covariancePropagation(mlp);
    expect(result.length).toBe(depth * width);
  });

  it('all values ≥ 0 (post-ReLU means must be non-negative)', () => {
    const mlp = sampleMLP(16, 4, 8);
    const result = covariancePropagation(mlp);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('all values are finite', () => {
    const mlp = sampleMLP(16, 4, 88);
    const result = covariancePropagation(mlp);
    for (let i = 0; i < result.length; i++) {
      expect(isFinite(result[i])).toBe(true);
    }
  });

  it('is deterministic (same mlp → same result)', () => {
    const mlp = sampleMLP(8, 3, 66);
    const r1 = covariancePropagation(mlp);
    const r2 = covariancePropagation(mlp);
    expect(Array.from(r1)).toEqual(Array.from(r2));
  });

  it('works for depth=1', () => {
    const mlp = sampleMLP(8, 1, 11);
    const result = covariancePropagation(mlp);
    expect(result.length).toBe(8);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('matches meanPropagation for a single layer (covariance starts as identity, so they should agree)', () => {
    const mlp = sampleMLP(8, 1, 77);
    const rMean = meanPropagation(mlp);
    const rCov  = covariancePropagation(mlp);
    // For a single layer with identity initial covariance, the two methods
    // should produce identical results because Cov_pre diagonal == diag(W^T W)
    // which matches the diagonal approximation used by meanPropagation.
    for (let j = 0; j < mlp.width; j++) {
      expect(rCov[j]).toBeCloseTo(rMean[j], 4);
    }
  });

  it('known single-layer case: identity weights give E[ReLU(z)] ≈ 1/sqrt(2pi)', () => {
    const width = 4;
    const weights = [new Float32Array(width * width)];
    for (let i = 0; i < width; i++) weights[0][i * width + i] = 1.0;
    const mlp = { width, depth: 1, weights };
    const result = covariancePropagation(mlp);
    const expected = 1.0 / Math.sqrt(2 * Math.PI);
    for (let j = 0; j < width; j++) {
      expect(result[j]).toBeCloseTo(expected, 3);
    }
  });

  it('close to ground truth for shallow network (width=16, depth=2, MSE < 0.01)', () => {
    const width = 16;
    const depth = 2;
    const mlp = sampleMLP(width, depth, 42);
    const { means: groundTruth } = outputStats(mlp, 50000, 0);
    const predicted = covariancePropagation(mlp);
    const error = mse(predicted, groundTruth);
    expect(error).toBeLessThan(0.01);
  });

  it('at least as accurate as meanPropagation for deep networks (width=16, depth=8)', () => {
    const width = 16;
    const depth = 8;
    const mlp = sampleMLP(width, depth, 123);

    // Ground truth: 50k samples
    const { means: groundTruth } = outputStats(mlp, 50000, 0);

    const predMean = meanPropagation(mlp);
    const predCov  = covariancePropagation(mlp);

    const mseMean = mse(predMean, groundTruth);
    const mseCov  = mse(predCov,  groundTruth);

    // covariancePropagation should be at most 10% worse than meanPropagation
    // (and typically better or comparable)
    expect(mseCov).toBeLessThanOrEqual(mseMean * 1.1);
  });
});
