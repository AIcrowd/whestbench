/**
 * math-utils.js — shared math utilities for mlp.js and estimators.js
 *
 * Exports:
 *   normalPdf(x)     — standard normal PDF
 *   normalCdf(x)     — standard normal CDF (Abramowitz & Stegun 26.2.17)
 *   boxMuller(rng)   — Box-Muller transform, returns [g1, g2]
 *   makeRng(seed)    — seedable xoshiro128** PRNG with SplitMix32 expansion
 */

const SQRT_2PI = Math.sqrt(2 * Math.PI);

/**
 * Standard normal PDF: exp(-x²/2) / √(2π)
 * @param {number} x
 * @returns {number}
 */
export function normalPdf(x) {
  return Math.exp(-0.5 * x * x) / SQRT_2PI;
}

/**
 * Standard normal CDF via Abramowitz & Stegun 26.2.17 rational approximation.
 * Maximum absolute error < 7.5e-8.
 * @param {number} x
 * @returns {number}
 */
export function normalCdf(x) {
  // A&S 26.2.17 coefficients
  const p = 0.2316419;
  const b1 = 0.319381530;
  const b2 = -0.356563782;
  const b3 = 1.781477937;
  const b4 = -1.821255978;
  const b5 = 1.330274429;

  const t = 1.0 / (1.0 + p * Math.abs(x));
  const poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))));
  const pdf = Math.exp(-0.5 * x * x) / SQRT_2PI;
  const cdf = 1.0 - pdf * poly;

  return x >= 0 ? cdf : 1.0 - cdf;
}

/**
 * SplitMix32 — used to expand a single seed into 4 independent state words.
 * @param {number} seed — 32-bit integer seed
 * @returns {Uint32Array} four 32-bit state words
 */
function splitMix32Expand(seed) {
  let s = seed >>> 0;
  const state = new Uint32Array(4);
  for (let i = 0; i < 4; i++) {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = (z ^ (z >>> 16)) >>> 0;
    z = Math.imul(z, 0x85ebca6b) >>> 0;
    z = (z ^ (z >>> 13)) >>> 0;
    z = Math.imul(z, 0xc2b2ae35) >>> 0;
    z = (z ^ (z >>> 16)) >>> 0;
    state[i] = z;
  }
  return state;
}

/**
 * Seedable PRNG using xoshiro128** algorithm with SplitMix32 seed expansion.
 * @param {number} seed — integer seed
 * @returns {{ random(): number, randInt(lo: number, hi: number): number }}
 */
export function makeRng(seed = 42) {
  const s = splitMix32Expand(seed);

  function next() {
    // xoshiro128** step
    const result = Math.imul(s[1], 5);
    const rotated = ((result << 7) | (result >>> 25)) >>> 0;
    const output = Math.imul(rotated, 9) >>> 0;

    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = ((s[3] << 11) | (s[3] >>> 21)) >>> 0;

    return output / 0x100000000;
  }

  return {
    /** Returns float in [0, 1) */
    random: next,
    /**
     * Returns integer in [lo, hi)
     * @param {number} lo — inclusive lower bound
     * @param {number} hi — exclusive upper bound
     */
    randInt(lo, hi) {
      return lo + Math.floor(next() * (hi - lo));
    },
  };
}

/**
 * Box-Muller transform — generates a pair of independent standard normal samples
 * from a uniform PRNG.
 * @param {{ random(): number }} rng — PRNG with .random() method
 * @returns {[number, number]} [g1, g2] — pair of standard normal samples
 */
export function boxMuller(rng) {
  const u1 = rng.random();
  const u2 = rng.random();
  // Avoid log(0): u1 is practically never 0 with a good PRNG, but clamp just in case
  const r = Math.sqrt(-2.0 * Math.log(u1 === 0 ? Number.EPSILON : u1));
  const theta = 2.0 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}
