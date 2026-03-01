/**
 * gateShapes.js — Gate classification for circuit visualization.
 *
 * Every gate computes: output = c + a·x + b·y + p·x·y
 * All gates use the SAME visual shape (rectangle).
 * Gate type is indicated by a subtle color tint on the border.
 */

/* Uniform gate dimensions */
export const GATE_W = 48;
export const GATE_H = 32;

/* Gate type → border color (all use same rectangle shape) */
const TYPE_COLORS = {
  and:      { stroke: "#EF4444", text: "#991B1B" },
  linear:   { stroke: "#3B82F6", text: "#1E40AF" },
  product:  { stroke: "#F59E0B", text: "#92400E" },
  constant: { stroke: "#9CA3AF", text: "#6B7280" },
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
 * Blue (-1) → White (0) → Red (+1)
 */
export function meanToColor(mean) {
  if (mean === null || mean === undefined) return null;
  const t = Math.max(-1, Math.min(1, mean));
  if (t < 0) {
    const s = 1 + t;
    const r = Math.round(59 + (255 - 59) * s);
    const g = Math.round(130 + (255 - 130) * s);
    const b = Math.round(246 + (255 - 246) * s);
    return `rgb(${r},${g},${b})`;
  } else {
    const r = 255;
    const g = Math.round(255 - (255 - 82) * t);
    const b = Math.round(255 - (255 - 77) * t);
    return `rgb(${r},${g},${b})`;
  }
}
