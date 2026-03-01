/**
 * gateShapes.js — Gate classification for circuit visualization.
 *
 * Every gate computes: output = c + a·x + b·y + p·x·y
 * All gates use the SAME visual shape (rectangle).
 *
 * Unified color palette anchored on AIcrowd coral #F0524D.
 */

/* Uniform gate dimensions */
export const GATE_W = 48;
export const GATE_H = 32;

/* Gate type → border color (all use same rectangle shape) */
const TYPE_COLORS = {
  and:      { stroke: "#F0524D", text: "#991B1B" },
  linear:   { stroke: "#94A3B8", text: "#475569" },
  product:  { stroke: "#F0524D", text: "#991B1B" },
  constant: { stroke: "#D1D5DB", text: "#6B7280" },
};

/**
 * Classify a gate by its coefficient pattern.
 */
export function classifyGate(layer, wireIndex) {
  const c = layer.const[wireIndex];
  const a = layer.firstCoeff[wireIndex];
  const b = layer.secondCoeff[wireIndex];
  const p = layer.productCoeff[wireIndex];

  const hasC = Math.abs(c) > 1e-6;
  const hasA = Math.abs(a) > 1e-6;
  const hasB = Math.abs(b) > 1e-6;
  const hasP = Math.abs(p) > 1e-6;

  const nonzero = [hasC, hasA, hasB, hasP].filter(Boolean).length;
  if (nonzero === 1) {
    if (hasA || hasB) return { type: "linear" };
    if (hasC) return { type: "constant" };
    if (hasP) return { type: "product" };
  }
  return { type: "and" };
}

export function gateColor(type) {
  return TYPE_COLORS[type] || TYPE_COLORS.and;
}

/**
 * Map a mean value in [-1,1] to a fill color.
 * Dark slate (-1) → White (0) → Coral (+1)
 *
 * Unified: uses AIcrowd coral #F0524D for positive,
 * and dark slate #334155 for negative.
 * No blue — avoids conflict with categorical/chart colors.
 */
export function meanToColor(mean) {
  if (mean === null || mean === undefined) return null;
  const t = Math.max(-1, Math.min(1, mean));
  if (t < 0) {
    // Dark slate (#334155) → White (#FFFFFF)
    const s = 1 + t; // 0 at -1, 1 at 0
    const r = Math.round(51 + (255 - 51) * s);
    const g = Math.round(65 + (255 - 65) * s);
    const b = Math.round(85 + (255 - 85) * s);
    return `rgb(${r},${g},${b})`;
  } else {
    // White (#FFFFFF) → Coral (#F0524D)
    const r = Math.round(255 - (255 - 240) * t);
    const g = Math.round(255 - (255 - 82) * t);
    const b = Math.round(255 - (255 - 77) * t);
    return `rgb(${r},${g},${b})`;
  }
}
