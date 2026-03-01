/**
 * gateShapes.js — Classify gates and define visual properties
 * for JointJS circuit rendering.
 *
 * Every gate computes: output = c + a·x + b·y + p·x·y
 * Visual shape depends on which coefficients are nonzero.
 */

/**
 * Classify a gate by its coefficient pattern.
 * Returns { type, label, shape, fill } for rendering.
 */
export function classifyGate(layer, wireIndex) {
  const c = layer.const[wireIndex];
  const a = layer.firstCoeff[wireIndex];
  const b = layer.secondCoeff[wireIndex];
  const p = layer.productCoeff[wireIndex];
  const xi = layer.first[wireIndex];
  const yi = layer.second[wireIndex];

  const hasC = Math.abs(c) > 1e-6;
  const hasA = Math.abs(a) > 1e-6;
  const hasB = Math.abs(b) > 1e-6;
  const hasP = Math.abs(p) > 1e-6;

  const nonzero = [hasC, hasA, hasB, hasP].filter(Boolean).length;

  // Simple gate: exactly one nonzero coefficient
  if (nonzero === 1) {
    if (hasA) {
      return {
        type: "linear",
        label: `${a > 0 ? "" : "−"}x${xi}`,
        shape: "triangle",
        color: "#3B82F6", // blue
      };
    }
    if (hasB) {
      return {
        type: "linear",
        label: `${b > 0 ? "" : "−"}y${yi}`,
        shape: "triangle",
        color: "#3B82F6",
      };
    }
    if (hasC) {
      return {
        type: "constant",
        label: c > 0 ? "+1" : "−1",
        shape: "square",
        color: "#9CA3AF", // gray
      };
    }
    if (hasP) {
      return {
        type: "product",
        label: `${p > 0 ? "" : "−"}x${xi}·y${yi}`,
        shape: "circle",
        color: "#F59E0B", // amber
      };
    }
  }

  // Complex gate: AND-like (all 4 coefficients)
  const signX = a > 0 ? "+" : "−";
  const signY = b > 0 ? "+" : "−";
  return {
    type: "and",
    label: `AND(${signX}x${xi},${signY}y${yi})`,
    shape: "dshape",
    color: "#F0524D", // coral (AND gates)
  };
}

/**
 * Gate SVG path generators for JointJS markup.
 */
export const GATE_PATHS = {
  // Triangle (buffer/passthrough) — pointing right
  triangle: "M 0 0 L 40 20 L 0 40 Z",
  // Square (constant)
  square: "M 0 0 L 40 0 L 40 40 L 0 40 Z",
  // Circle (product)
  circle: "M 20 0 A 20 20 0 1 1 20 40 A 20 20 0 1 1 20 0 Z",
  // D-shape (AND gate) — flat left, curved right
  dshape: "M 0 0 L 20 0 Q 40 0 40 20 Q 40 40 20 40 L 0 40 Z",
};

/**
 * Color scale: map a mean value in [-1, 1] to a fill color.
 * Blue (-1) → White (0) → Red (+1)
 */
export function meanToColor(mean) {
  if (mean === null || mean === undefined) return "#E5E7EB"; // gray when no data
  const t = Math.max(-1, Math.min(1, mean));
  if (t < 0) {
    // Blue to white
    const s = 1 + t; // 0 at -1, 1 at 0
    const r = Math.round(59 + (255 - 59) * s);
    const g = Math.round(130 + (255 - 130) * s);
    const b = Math.round(246 + (255 - 246) * s);
    return `rgb(${r},${g},${b})`;
  } else {
    // White to red
    const r = 255;
    const g = Math.round(255 - (255 - 82) * t);
    const b = Math.round(255 - (255 - 77) * t);
    return `rgb(${r},${g},${b})`;
  }
}
