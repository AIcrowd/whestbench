/**
 * gateShapes.js — Gate classification & visualization constants.
 *
 * Every gate computes: output = c + a·x + b·y + p·x·y
 * All gates use the SAME visual shape (rectangle).
 *
 * The classifier maps coefficient patterns to one of 9 named boolean
 * function families, each with a unicode symbol and display color.
 */

/* === Tunable visualization params (single source of truth) === */
export const GATE_W         = 20;   // gate body width
export const GATE_H         = 32;   // gate body height
export const GATE_OPACITY   = 0.3;  // gate body fill opacity — ghosted
export const WIRE_PORT_R    = 11;   // output port circle radius
export const INPUT_DOT_R    = 5;    // input/output wire endpoint dot radius

/**
 * GATE_TYPES — canonical gate definitions.
 * Each type gets a unicode symbol and a display color.
 */
export const GATE_TYPES = {
  BUF:   { symbol: "▷",  color: "#6366F1" },  // indigo
  NOT:   { symbol: "▷○", color: "#8B5CF6" },  // violet
  CONST: { symbol: "■",  color: "#9CA3AF" },  // gray
  XOR:   { symbol: "⊕",  color: "#F59E0B" },  // amber
  XNOR:  { symbol: "⊙",  color: "#D97706" },  // dark amber
  AND:   { symbol: "∧",  color: "#10B981" },  // emerald
  NAND:  { symbol: "∧̄",  color: "#059669" },  // dark emerald
  OR:    { symbol: "∨",  color: "#3B82F6" },  // blue
  NOR:   { symbol: "∨̄",  color: "#2563EB" },  // dark blue
};

/**
 * Classify a gate by its coefficient pattern into one of the boolean
 * function families.
 *
 * @param {object} layer  — layer object with typed arrays
 * @param {number} wireIndex — wire index within the layer
 * @returns {{ type: string, label: string, symbol: string, color: string }}
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

  // --- Simple gates: exactly one nonzero coefficient ---
  if (!hasP && !hasC && (hasA !== hasB)) {
    // Single linear term: buffer or NOT
    if (hasA) {
      const t = a > 0 ? "BUF" : "NOT";
      return { type: t, label: a > 0 ? "x" : "−x", ...GATE_TYPES[t] };
    }
    const t = b > 0 ? "BUF" : "NOT";
    return { type: t, label: b > 0 ? "y" : "−y", ...GATE_TYPES[t] };
  }

  if (hasC && !hasA && !hasB && !hasP) {
    return { type: "CONST", label: c > 0 ? "+1" : "−1", ...GATE_TYPES.CONST };
  }

  if (!hasC && !hasA && !hasB && hasP) {
    const t = p > 0 ? "XNOR" : "XOR";
    return { type: t, label: t === "XNOR" ? "XNOR(x, y)" : "XOR(x, y)", ...GATE_TYPES[t] };
  }

  // --- Complex gates: AND family (affine-bilinear) ---
  // Gate computes: coeff * (−1 + xc·x + yc·y + xc·yc·xy) = ±AND(±x, ±y)
  const xSign = (a / p) > 0 ? "" : "−";
  const ySign = (b > 0) === (p > 0) ? "" : "−";
  const outerPositive = (c < 0) === (p > 0);

  const xNeg = xSign === "−";
  const yNeg = ySign === "−";

  let type, label;
  if (outerPositive) {
    if (!xNeg && !yNeg) {
      type = "AND";  label = "AND(x, y)";
    } else if (xNeg && yNeg) {
      type = "NOR";  label = "NOR(x, y)";
    } else {
      type = "AND";  label = `AND(${xSign}x, ${ySign}y)`;
    }
  } else {
    if (!xNeg && !yNeg) {
      type = "NAND"; label = "NAND(x, y)";
    } else if (xNeg && yNeg) {
      type = "OR";   label = "OR(x, y)";
    } else {
      type = "NAND"; label = `NAND(${xSign}x, ${ySign}y)`;
    }
  }

  return { type, label, ...GATE_TYPES[type] };
}

/**
 * Get display properties for a gate type key.
 * @param {string} type — key from GATE_TYPES (e.g. "AND", "XOR")
 * @returns {{ symbol: string, color: string }}
 */
export function gateColor(type) {
  return GATE_TYPES[type] || GATE_TYPES.AND;
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
