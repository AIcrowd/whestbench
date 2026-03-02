/**
 * CanvasTooltip — Reusable portal-based tooltip for canvas charts.
 * Renders children near the cursor with viewport edge-flipping.
 * Used by GateStats, ActivationRibbon, WireStats, CoeffHistograms.
 */
import { createPortal } from "react-dom";

export default function CanvasTooltip({ visible, pageX, pageY, children }) {
  if (!visible) return null;

  const vpW = typeof window !== "undefined" ? window.innerWidth : 1920;
  const vpH = typeof window !== "undefined" ? window.innerHeight : 1080;
  const TOOLTIP_W = 260;
  const TOOLTIP_H = 180;

  const left = pageX + TOOLTIP_W + 20 > vpW
    ? pageX - TOOLTIP_W - 12
    : pageX + 14;
  const top = pageY + TOOLTIP_H + 20 > vpH
    ? Math.max(8, pageY - TOOLTIP_H)
    : Math.max(8, pageY - 10);

  return createPortal(
    <div className="canvas-data-tooltip" style={{ left, top }}>
      {children}
    </div>,
    document.body
  );
}
