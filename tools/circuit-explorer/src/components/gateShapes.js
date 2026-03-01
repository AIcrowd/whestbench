/**
 * gateShapes.js — Custom JointJS shape definitions for circuit gates.
 *
 * Every gate computes: output = c + a·x + b·y + p·x·y
 * Visual shape depends on which coefficients are nonzero.
 *
 * Gate types:
 *   AND  →  D-shape (flat left wall, curved right — IEEE AND notation)
 *   Linear  →  Triangle pointing right (buffer/amplifier)
 *   Product →  Circle with × cross (multiplier)
 *   Constant →  Square with flat line (fixed source)
 */
import { dia } from "@joint/core";

/* ============================================================ */
/*  1. Custom JointJS shape definitions                          */
/* ============================================================ */

/**
 * AND gate — D-shape (IEEE Std 91-1984).
 * Flat left wall with a curved right side.
 */
const GateAND = dia.Element.define(
  "circuit.GateAND",
  {
    size: { width: 64, height: 38 },
    attrs: {
      body: {
        d: "M 0 0 L 32 0 C 64 0 64 38 32 38 L 0 38 Z",
        fill: "#FEE2E2",
        stroke: "#F0524D",
        strokeWidth: 2,
      },
      label: {
        text: "AND",
        x: 28,
        y: 19,
        textAnchor: "middle",
        textVerticalAnchor: "middle",
        fontSize: 9,
        fontFamily: "'IBM Plex Mono', monospace",
        fill: "#991B1B",
      },
      // Input ports (decorative)
      portIn1: {
        d: "M -6 12 L 0 12",
        stroke: "#F0524D",
        strokeWidth: 1.5,
        fill: "none",
      },
      portIn2: {
        d: "M -6 26 L 0 26",
        stroke: "#F0524D",
        strokeWidth: 1.5,
        fill: "none",
      },
      // Output port
      portOut: {
        d: "M 56 19 L 64 19",
        stroke: "#F0524D",
        strokeWidth: 1.5,
        fill: "none",
      },
    },
  },
  {
    markup: [
      { tagName: "path", selector: "body" },
      { tagName: "path", selector: "portIn1" },
      { tagName: "path", selector: "portIn2" },
      { tagName: "path", selector: "portOut" },
      { tagName: "text", selector: "label" },
    ],
  }
);

/**
 * Linear gate — Triangle pointing right (buffer/amplifier).
 */
const GateLinear = dia.Element.define(
  "circuit.GateLinear",
  {
    size: { width: 64, height: 38 },
    attrs: {
      body: {
        d: "M 0 0 L 56 19 L 0 38 Z",
        fill: "#DBEAFE",
        stroke: "#3B82F6",
        strokeWidth: 2,
        strokeLinejoin: "round",
      },
      label: {
        text: "x",
        x: 20,
        y: 19,
        textAnchor: "middle",
        textVerticalAnchor: "middle",
        fontSize: 9,
        fontFamily: "'IBM Plex Mono', monospace",
        fill: "#1E40AF",
      },
      portIn: {
        d: "M -6 19 L 0 19",
        stroke: "#3B82F6",
        strokeWidth: 1.5,
        fill: "none",
      },
      portOut: {
        d: "M 56 19 L 64 19",
        stroke: "#3B82F6",
        strokeWidth: 1.5,
        fill: "none",
      },
    },
  },
  {
    markup: [
      { tagName: "path", selector: "body" },
      { tagName: "path", selector: "portIn" },
      { tagName: "path", selector: "portOut" },
      { tagName: "text", selector: "label" },
    ],
  }
);

/**
 * Product gate — Circle with cross marks (×  multiplier).
 */
const GateProduct = dia.Element.define(
  "circuit.GateProduct",
  {
    size: { width: 38, height: 38 },
    attrs: {
      body: {
        d: "M 19 0 A 19 19 0 1 1 19 38 A 19 19 0 1 1 19 0 Z",
        fill: "#FEF3C7",
        stroke: "#F59E0B",
        strokeWidth: 2,
      },
      cross1: {
        d: "M 12 12 L 26 26",
        stroke: "#B45309",
        strokeWidth: 1.5,
        fill: "none",
      },
      cross2: {
        d: "M 26 12 L 12 26",
        stroke: "#B45309",
        strokeWidth: 1.5,
        fill: "none",
      },
      label: {
        text: "×",
        x: 19,
        y: 19,
        textAnchor: "middle",
        textVerticalAnchor: "middle",
        fontSize: 9,
        fontFamily: "'IBM Plex Mono', monospace",
        fill: "#92400E",
      },
      portIn1: {
        d: "M -6 12 L 3 12",
        stroke: "#F59E0B",
        strokeWidth: 1.5,
        fill: "none",
      },
      portIn2: {
        d: "M -6 26 L 3 26",
        stroke: "#F59E0B",
        strokeWidth: 1.5,
        fill: "none",
      },
      portOut: {
        d: "M 35 19 L 44 19",
        stroke: "#F59E0B",
        strokeWidth: 1.5,
        fill: "none",
      },
    },
  },
  {
    markup: [
      { tagName: "path", selector: "body" },
      { tagName: "path", selector: "cross1" },
      { tagName: "path", selector: "cross2" },
      { tagName: "path", selector: "portIn1" },
      { tagName: "path", selector: "portIn2" },
      { tagName: "path", selector: "portOut" },
      { tagName: "text", selector: "label" },
    ],
  }
);

/**
 * Constant gate — Square with a horizontal line (DC source).
 */
const GateConstant = dia.Element.define(
  "circuit.GateConstant",
  {
    size: { width: 42, height: 38 },
    attrs: {
      body: {
        d: "M 0 0 L 42 0 L 42 38 L 0 38 Z",
        fill: "#F3F4F6",
        stroke: "#9CA3AF",
        strokeWidth: 2,
      },
      // Flat line symbol (DC)
      dcLine: {
        d: "M 12 19 L 30 19",
        stroke: "#6B7280",
        strokeWidth: 2,
        fill: "none",
      },
      label: {
        text: "+1",
        x: 21,
        y: 30,
        textAnchor: "middle",
        textVerticalAnchor: "middle",
        fontSize: 8,
        fontFamily: "'IBM Plex Mono', monospace",
        fill: "#6B7280",
      },
      portOut: {
        d: "M 42 19 L 50 19",
        stroke: "#9CA3AF",
        strokeWidth: 1.5,
        fill: "none",
      },
    },
  },
  {
    markup: [
      { tagName: "path", selector: "body" },
      { tagName: "path", selector: "dcLine" },
      { tagName: "path", selector: "portOut" },
      { tagName: "text", selector: "label" },
    ],
  }
);

/* ============================================================ */
/*  2. Shape registry — map shape name to constructor            */
/* ============================================================ */
export const GATE_CONSTRUCTORS = {
  dshape: GateAND,
  triangle: GateLinear,
  circle: GateProduct,
  square: GateConstant,
};

/* ============================================================ */
/*  3. Gate classifier                                           */
/* ============================================================ */

/**
 * Classify a gate by its coefficient pattern.
 * Returns { type, label, shape, color } for rendering.
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

/* ============================================================ */
/*  4. Color scale                                               */
/* ============================================================ */

/**
 * Color scale: map a mean value in [-1, 1] to a fill color.
 * Blue (-1) → White (0) → Red (+1)
 */
export function meanToColor(mean) {
  if (mean === null || mean === undefined) return "#E5E7EB";
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
