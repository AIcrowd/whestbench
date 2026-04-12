import { describe, it, expect } from 'vitest';
import { normalPdf, normalCdf, makeRng, boxMuller } from '../math-utils.js';

describe('normalPdf', () => {
  it('normalPdf(0) ≈ 0.3989422804', () => {
    expect(normalPdf(0)).toBeCloseTo(0.3989422804, 6);
  });

  it('normalPdf(1) ≈ 0.2419707245', () => {
    expect(normalPdf(1)).toBeCloseTo(0.2419707245, 6);
  });

  it('is symmetric: normalPdf(-x) === normalPdf(x)', () => {
    expect(normalPdf(-1)).toBeCloseTo(normalPdf(1), 10);
    expect(normalPdf(-2)).toBeCloseTo(normalPdf(2), 10);
    expect(normalPdf(-0.5)).toBeCloseTo(normalPdf(0.5), 10);
  });
});

describe('normalCdf', () => {
  it('normalCdf(0) ≈ 0.5', () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 8);
  });

  it('normalCdf(1) ≈ 0.8413447461', () => {
    expect(normalCdf(1)).toBeCloseTo(0.8413447461, 6);
  });

  it('normalCdf(-1) ≈ 0.1586552539', () => {
    expect(normalCdf(-1)).toBeCloseTo(0.1586552539, 6);
  });

  it('normalCdf(-6) ≈ 0', () => {
    expect(normalCdf(-6)).toBeCloseTo(0, 8);
  });

  it('normalCdf(6) ≈ 1', () => {
    expect(normalCdf(6)).toBeCloseTo(1, 8);
  });
});

describe('makeRng', () => {
  it('is deterministic with the same seed', () => {
    const rng1 = makeRng(42);
    const rng2 = makeRng(42);
    const samples1 = Array.from({ length: 20 }, () => rng1.random());
    const samples2 = Array.from({ length: 20 }, () => rng2.random());
    expect(samples1).toEqual(samples2);
  });

  it('produces different sequences with different seeds', () => {
    const rng1 = makeRng(42);
    const rng2 = makeRng(99);
    const samples1 = Array.from({ length: 10 }, () => rng1.random());
    const samples2 = Array.from({ length: 10 }, () => rng2.random());
    expect(samples1).not.toEqual(samples2);
  });

  it('random() returns values in [0, 1)', () => {
    const rng = makeRng(1);
    for (let i = 0; i < 100; i++) {
      const v = rng.random();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('randInt(lo, hi) returns integers in [lo, hi)', () => {
    const rng = makeRng(7);
    for (let i = 0; i < 100; i++) {
      const v = rng.randInt(0, 10);
      expect(Number.isInteger(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(10);
    }
  });
});

describe('boxMuller', () => {
  it('returns an array of length 2', () => {
    const rng = makeRng(1);
    const result = boxMuller(rng);
    expect(result).toHaveLength(2);
  });

  it('produces approximately standard normal samples (mean ≈ 0, variance ≈ 1) over 10k samples', () => {
    const rng = makeRng(123);
    const samples = [];
    for (let i = 0; i < 5000; i++) {
      const [g1, g2] = boxMuller(rng);
      samples.push(g1, g2);
    }
    // mean
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(mean).toBeCloseTo(0, 1); // within 0.1

    // variance
    const variance = samples.reduce((acc, x) => acc + (x - mean) ** 2, 0) / samples.length;
    expect(variance).toBeCloseTo(1, 1); // within 0.1
  });
});
