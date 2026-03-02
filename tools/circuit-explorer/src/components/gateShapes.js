/**
 * gateShapes.js — Gate classification & visualization constants.
 *
 * Every gate computes: output = c + a·x + b·y + p·x·y
 * All gates use the SAME visual shape (rectangle).
 *
 * Gate types are stored at creation time in circuit.js, not inferred.
 */

/* === Tunable visualization params (single source of truth) === */
export const GATE_W         = 20;   // gate body width
export const GATE_H         = 32;   // gate body height
export const GATE_OPACITY   = 0.3;  // gate body fill opacity — ghosted
export const WIRE_PORT_R    = 11;   // output port circle radius
export const INPUT_DOT_R    = 5;    // input/output wire endpoint dot radius
export const GATE_TYPE_FONT = "'Bradley Hand', Caveat, cursive"; // gate type label font

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
 * Look up the gate type info for a gate.
 * Gate types are stored at creation time in circuit.js (layer.gateType[i]).
 *
 * @param {object} layer  — layer object from circuit.gates[]
 * @param {number} wireIndex — wire index within the layer
 * @returns {{ type: string, label: string, symbol: string, color: string }}
 */
export function classifyGate(layer, wireIndex) {
  const type = layer.gateType?.[wireIndex] || "AND";
  const label = layer.gateLabel?.[wireIndex] || "?";
  const info = GATE_TYPES[type] || GATE_TYPES.AND;
  return { type, label, ...info };
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
