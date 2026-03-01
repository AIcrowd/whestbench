/**
 * gateShapes.js — Gate classification + visual properties for JointJS circuit.
 *
 * Every gate computes: output = c + a·x + b·y + p·x·y
 * Visual shape depends on which coefficients are nonzero.
 *
 * All gates use the SAME JointJS size (GATE_W × GATE_H) so columns align.
 * The SVG path inside varies to convey gate type.
 * Ports are added at the graph-component level (not in shape definition)
 * so we can use JointJS port groups ('in' on left, 'out' on right).
 */

/* ============================================================ */
/*  Uniform gate dimensions                                      */
/* ============================================================ */
export const GATE_W = 52;
export const GATE_H = 36;

/* ============================================================ */
/*  Gate type info                                                */
/* ============================================================ */

/**
 * Gate type visual config: SVG path (inside GATE_W × GATE_H box),
 * fill color, stroke color.
 */
export const GATE_TYPES = {
  and: {
    // D-shape: flat left, curved right (IEEE AND)
    path: `M 0 0 L ${GATE_W * 0.5} 0 Q ${GATE_W} 0 ${GATE_W} ${GATE_H / 2} Q ${GATE_W} ${GATE_H} ${GATE_W * 0.5} ${GATE_H} L 0 ${GATE_H} Z`,
    fill: "#FEE2E2",
    stroke: "#EF4444",
    textColor: "#991B1B",
  },
  linear: {
    // Triangle pointing right (buffer)
    path: `M 0 0 L ${GATE_W} ${GATE_H / 2} L 0 ${GATE_H} Z`,
    fill: "#DBEAFE",
    stroke: "#3B82F6",
    textColor: "#1E40AF",
  },
  product: {
    // Circle centered in the box
    path: `M ${GATE_W / 2} 0 A ${Math.min(GATE_W, GATE_H) / 2} ${Math.min(GATE_W, GATE_H) / 2} 0 1 1 ${GATE_W / 2} ${GATE_H} A ${Math.min(GATE_W, GATE_H) / 2} ${Math.min(GATE_W, GATE_H) / 2} 0 1 1 ${GATE_W / 2} 0 Z`,
    fill: "#FEF3C7",
    stroke: "#F59E0B",
    textColor: "#92400E",
  },
  constant: {
    // Square
    path: `M 0 0 L ${GATE_W} 0 L ${GATE_W} ${GATE_H} L 0 ${GATE_H} Z`,
    fill: "#F3F4F6",
    stroke: "#9CA3AF",
    textColor: "#6B7280",
  },
};

/* ============================================================ */
/*  Gate classifier                                              */
/* ============================================================ */

/**
 * Classify a gate by its coefficient pattern.
 * Returns { type, label } for rendering.
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
    if (hasA) return { type: "linear", label: `${a > 0 ? "" : "−"}x` };
    if (hasB) return { type: "linear", label: `${b > 0 ? "" : "−"}y` };
    if (hasC) return { type: "constant", label: c > 0 ? "+1" : "−1" };
    if (hasP) return { type: "product", label: "×" };
  }

  return { type: "and", label: "∧" };
}

/* ============================================================ */
/*  Color scale                                                  */
/* ============================================================ */

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
