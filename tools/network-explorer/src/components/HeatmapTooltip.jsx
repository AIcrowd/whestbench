/**
 * HeatmapTooltip — Reusable hover overlay for heatmaps.
 * Renders:
 *   - A crosshair overlay canvas on top of the heatmap
 *   - A floating tooltip (via portal) showing neuron/layer/value
 *   - A magnifier canvas showing zoomed-in neighborhood
 *
 * The tooltip is rendered via React portal to document.body so it
 * isn't clipped by parent containers with overflow: hidden.
 *
 * Props:
 *   width, height — CSS dimensions of the heatmap canvas
 *   n — number of neurons (rows)
 *   d — number of layers (columns)
 *   getData(layer, neuron) — returns the numeric value for the cell
 *   getColor(layer, neuron) — returns [r, g, b] for the cell
 *   valueLabel — label for the value (e.g. "σ" or "|error|")
 *   formatValue — optional (v) => string formatter
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const ZOOM_RADIUS = 5;        // cells around cursor to show
const ZOOM_CELL_PX = 14;     // base px per cell in magnifier
const ZOOM_MAX_SIZE = 200;    // max canvas dimension in px

export default function HeatmapTooltip({
  width, height, n, d,
  getData, getColor, valueLabel = "value",
  formatValue = (v) => v.toFixed(4),
  showZoom = true,
}) {
  const overlayRef = useRef(null);
  const zoomCanvasRef = useRef(null);
  const lastCellRef = useRef(null);
  const rafRef = useRef(null);
  const [hovered, setHovered] = useState(null);

  const cellW = width / d;
  const cellH = height / n;

  // Draw crosshair on overlay canvas
  const drawCrosshair = useCallback((layer, neuron) => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    if (layer === null || neuron === null) return;

    // Semi-transparent crosshair lines
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;

    const lx = layer * cellW + cellW / 2;
    ctx.beginPath();
    ctx.moveTo(lx, 0);
    ctx.lineTo(lx, height);
    ctx.stroke();

    const wy = neuron * cellH + cellH / 2;
    ctx.beginPath();
    ctx.moveTo(0, wy);
    ctx.lineTo(width, wy);
    ctx.stroke();

    // Cell highlight
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(layer * cellW, neuron * cellH, cellW, cellH);
  }, [width, height, cellW, cellH]);

  // Compute zoom cell sizes proportional to main heatmap aspect ratio
  const zoomCellW = (() => {
    if (!width || !height || !d || !n) return ZOOM_CELL_PX;
    const mainRatio = (width / d) / (height / n); // cellW / cellH
    if (mainRatio >= 1) {
      // cells wider than tall — scale up width, keep height at base
      return Math.min(ZOOM_CELL_PX * mainRatio, ZOOM_MAX_SIZE / (ZOOM_RADIUS * 2 + 1));
    }
    return ZOOM_CELL_PX;
  })();
  const zoomCellH = (() => {
    if (!width || !height || !d || !n) return ZOOM_CELL_PX;
    const mainRatio = (width / d) / (height / n);
    if (mainRatio < 1) {
      // cells taller than wide — scale up height, keep width at base
      return Math.min(ZOOM_CELL_PX / mainRatio, ZOOM_MAX_SIZE / (ZOOM_RADIUS * 2 + 1));
    }
    return ZOOM_CELL_PX;
  })();
  const zoomW = (ZOOM_RADIUS * 2 + 1) * zoomCellW;
  const zoomH = (ZOOM_RADIUS * 2 + 1) * zoomCellH;

  // Draw zoom magnifier
  const drawZoom = useCallback((layer, neuron) => {
    const canvas = zoomCanvasRef.current;
    if (!canvas || !getColor) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(zoomW * dpr);
    canvas.height = Math.round(zoomH * dpr);
    canvas.style.width = `${Math.round(zoomW)}px`;
    canvas.style.height = `${Math.round(zoomH)}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    for (let dy = -ZOOM_RADIUS; dy <= ZOOM_RADIUS; dy++) {
      for (let dx = -ZOOM_RADIUS; dx <= ZOOM_RADIUS; dx++) {
        const l = layer + dx;
        const w = neuron + dy;
        const px = (dx + ZOOM_RADIUS) * zoomCellW;
        const py = (dy + ZOOM_RADIUS) * zoomCellH;

        if (l >= 0 && l < d && w >= 0 && w < n) {
          const [r, g, b] = getColor(l, w);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
        } else {
          ctx.fillStyle = "#1E293B";
        }
        ctx.fillRect(px, py, zoomCellW, zoomCellH);

        // Grid lines
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(px, py, zoomCellW, zoomCellH);
      }
    }

    // Highlight center cell
    const cx = ZOOM_RADIUS * zoomCellW;
    const cy = ZOOM_RADIUS * zoomCellH;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.strokeRect(cx, cy, zoomCellW, zoomCellH);
  }, [n, d, getColor, zoomCellW, zoomCellH, zoomW, zoomH]);

  // Set up overlay canvas dimensions
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }, [width, height]);

  // Mouse move — RAF-debounced, uses page coordinates for portal positioning
  const handleMouseMove = useCallback((e) => {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const canvas = overlayRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const layer = Math.floor(mx / cellW);
      const neuron = Math.floor(my / cellH);
      const key = `${layer},${neuron}`;

      if (key === lastCellRef.current) return;
      lastCellRef.current = key;

      if (layer >= 0 && layer < d && neuron >= 0 && neuron < n) {
        drawCrosshair(layer, neuron);
        if (showZoom) drawZoom(layer, neuron);
        const val = getData(layer, neuron);
        setHovered({
          neuron, layer, val,
          // Use page coordinates (not relative to container) for portal positioning
          pageX: e.pageX,
          pageY: e.pageY,
        });
      } else {
        drawCrosshair(null, null);
        setHovered(null);
      }
    });
  }, [cellW, cellH, n, d, drawCrosshair, drawZoom, getData, showZoom]);

  const handleMouseLeave = useCallback(() => {
    lastCellRef.current = null;
    drawCrosshair(null, null);
    setHovered(null);
  }, [drawCrosshair]);

  if (!width || !height) return null;

  // Position tooltip near cursor, flipping if near viewport edges
  const vpW = typeof window !== "undefined" ? window.innerWidth : 1920;
  const tipWidth = showZoom ? Math.round(zoomW) + 40 : 180;
  const tooltipLeft = hovered
    ? (hovered.pageX + tipWidth + 30 > vpW
      ? hovered.pageX - tipWidth - 24
      : hovered.pageX + 16)
    : 0;
  const tooltipTop = hovered ? Math.max(8, hovered.pageY - 20) : 0;

  return (
    <>
      <canvas
        ref={overlayRef}
        className="heatmap-overlay-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      {hovered && createPortal(
        <div
          className="canvas-data-tooltip"
          style={{ left: tooltipLeft, top: tooltipTop }}
        >
          <div className="canvas-tip-header">
            Neuron <span className="layer-num">{hovered.neuron}</span>
            {" · "}
            Layer <span className="layer-num">{hovered.layer}</span>
          </div>
          <div className="canvas-tip-rows">
            <div className="canvas-tip-row">
              <span className="canvas-tip-label">{valueLabel}</span>
              <span className="canvas-tip-value">{formatValue(hovered.val)}</span>
            </div>
          </div>
          {showZoom && <canvas ref={zoomCanvasRef} className="heatmap-zoom-canvas" style={{ margin: "0 8px 8px" }} />}
        </div>,
        document.body
      )}
    </>
  );
}
