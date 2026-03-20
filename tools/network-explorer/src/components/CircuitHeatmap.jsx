/**
 * CircuitHeatmap — Canvas-rendered wires × layers heatmap
 * for large circuits (n×d > 4096).
 * Shows wire means as a color grid, with hover triggering a detail overlay.
 *
 * Performance notes:
 * - Uses a main canvas for the heatmap grid (redrawn only when data changes)
 * - Uses a separate overlay canvas for crosshair (redrawn on mousemove — cheap)
 * - Debounces hover: RAF-gated, only updates when the hovered cell changes
 * - Resolution cap: sub-pixel cells are rendered at reduced resolution
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { perfEnd, perfStart } from "../perf";
import GateDetailOverlay from "./GateDetailOverlay";

export default function CircuitHeatmap({ circuit, means, activeLayer, onLayerClick }) {
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const containerRef = useRef(null);
  const [hovered, setHovered] = useState(null); // { wire, layer, x, y }
  const [dims, setDims] = useState({ cellW: 0, cellH: 0 });
  const lastCellRef = useRef(null);
  const rafRef = useRef(null);

  const n = circuit.n;
  const d = circuit.d;

  // Max height for the heatmap — fits nicely in the viewport
  const MAX_HEIGHT = 500;

  // Render heatmap to canvas
  useEffect(() => {
    perfStart('heatmap-paint');
    const canvas = canvasRef.current;
    const overlay = overlayCanvasRef.current;
    const container = containerRef.current;
    if (!canvas || !overlay || !container) return;

    const rect = container.getBoundingClientRect();
    const width = Math.floor(rect.width);
    const height = Math.min(MAX_HEIGHT, Math.max(200, Math.floor(n * 2)));

    const dpr = window.devicePixelRatio || 1;
    const rawCellW = width / d;
    const rawCellH = height / n;
    const renderScale = (rawCellW < 1 || rawCellH < 1)
      ? Math.max(1, Math.min(dpr, 2))
      : dpr;

    const canvasW = Math.round(width * renderScale);
    const canvasH = Math.round(height * renderScale);

    canvas.width = canvasW;
    canvas.height = canvasH;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    overlay.width = width * dpr;
    overlay.height = height * dpr;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");

    const cellW = width / d;
    const cellH = height / n;
    setDims({ cellW, cellH, width, height });

    // ── putImageData path: write RGBA directly, 1 call instead of 262k ──
    const imgData = ctx.createImageData(canvasW, canvasH);
    const pixels = imgData.data;

    for (let py = 0; py < canvasH; py++) {
      const wire = Math.floor((py / canvasH) * n);
      for (let px = 0; px < canvasW; px++) {
        const layer = Math.floor((px / canvasW) * d);
        const idx = (py * canvasW + px) * 4;
        const mean = means && means[layer] ? means[layer][wire] : null;

        if (mean !== null && mean !== undefined) {
          const t = Math.max(-1, Math.min(1, mean));
          if (t < 0) {
            const s = 1 + t;
            pixels[idx]     = 51 + (204 * s) | 0;
            pixels[idx + 1] = 65 + (190 * s) | 0;
            pixels[idx + 2] = 85 + (170 * s) | 0;
          } else {
            pixels[idx]     = 255 - (15 * t) | 0;
            pixels[idx + 1] = 255 - (173 * t) | 0;
            pixels[idx + 2] = 255 - (178 * t) | 0;
          }
        } else {
          pixels[idx]     = 229;
          pixels[idx + 1] = 231;
          pixels[idx + 2] = 235;
        }
        pixels[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Grid lines (only if cells are big enough to see)
    if (cellW > 3 && cellH > 3) {
      ctx.scale(renderScale, renderScale);
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 0.5;
      for (let l = 0; l <= d; l++) {
        ctx.beginPath();
        ctx.moveTo(l * cellW, 0);
        ctx.lineTo(l * cellW, height);
        ctx.stroke();
      }
      for (let w = 0; w <= n; w++) {
        ctx.beginPath();
        ctx.moveTo(0, w * cellH);
        ctx.lineTo(width, w * cellH);
        ctx.stroke();
      }
    }

    // Axis labels
    if (!ctx.getTransform || ctx.getTransform().a === 1) {
      ctx.scale(renderScale, renderScale);
    }
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    const labelStep = Math.max(1, Math.floor(d / 10));
    for (let l = 0; l < d; l += labelStep) {
      ctx.fillText(`${l}`, l * cellW + cellW / 2, height - 2);
    }
    perfEnd('heatmap-paint');
  }, [circuit, means, n, d]);

  // Draw crosshair + activeLayer column on overlay canvas
  const drawCrosshair = useCallback((layer, wire) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay || !dims.cellW) return;
    const dpr = window.devicePixelRatio || 1;
    const ctx = overlay.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, dims.width, dims.height);

    // Active layer column highlight
    if (activeLayer !== undefined && activeLayer !== null) {
      ctx.fillStyle = "rgba(240, 82, 77, 0.12)";
      ctx.fillRect(activeLayer * dims.cellW, 0, dims.cellW, dims.height);
      ctx.strokeStyle = "rgba(240, 82, 77, 0.5)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(activeLayer * dims.cellW, 0, dims.cellW, dims.height);
    }



    if (layer === null || wire === null) return;

    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 1;

    // Vertical line (layer column)
    const lx = layer * dims.cellW + dims.cellW / 2;
    ctx.beginPath();
    ctx.moveTo(lx, 0);
    ctx.lineTo(lx, dims.height);
    ctx.stroke();

    // Horizontal line (wire row)
    const wy = wire * dims.cellH + dims.cellH / 2;
    ctx.beginPath();
    ctx.moveTo(0, wy);
    ctx.lineTo(dims.width, wy);
    ctx.stroke();

    // Cell highlight border
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(
      layer * dims.cellW,
      wire * dims.cellH,
      dims.cellW,
      dims.cellH
    );
  }, [dims, activeLayer, d]);

  // Redraw overlay when activeLayer changes
  useEffect(() => {
    drawCrosshair(null, null);
  }, [activeLayer, drawCrosshair]);

  // Click handler — toggle activeLayer
  const handleClick = useCallback(
    (e) => {
      if (!dims.cellW) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const layer = Math.floor((e.clientX - rect.left) / dims.cellW);
      if (layer >= 0 && layer < d) {
        onLayerClick?.(layer === activeLayer ? undefined : layer);
      }
    },
    [dims, d, activeLayer, onLayerClick]
  );

  // Debounced mouse move — RAF-gated, only updates when cell changes
  const handleMouseMove = useCallback(
    (e) => {
      if (rafRef.current) return; // already scheduled
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        if (!dims.cellW) return;
        const rect = canvasRef.current.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const layer = Math.floor(mx / dims.cellW);
        const wire = Math.floor(my / dims.cellH);
        const cellKey = `${layer},${wire}`;

        if (cellKey === lastCellRef.current) return; // same cell, skip
        lastCellRef.current = cellKey;

        if (layer >= 0 && layer < d && wire >= 0 && wire < n) {
          drawCrosshair(layer, wire);
          setHovered({
            wire,
            layer,
            x: e.clientX - containerRef.current.getBoundingClientRect().left,
            y: e.clientY - containerRef.current.getBoundingClientRect().top,
          });
        } else {
          drawCrosshair(null, null);
          setHovered(null);
        }
      });
    },
    [dims, n, d, drawCrosshair]
  );

  const handleMouseLeave = useCallback(() => {
    lastCellRef.current = null;
    drawCrosshair(null, null);
    setHovered(null);
  }, [drawCrosshair]);

  return (
    <div className="panel circuit-heatmap" ref={containerRef}>
      <h2>
        Circuit Structure
        <span className="mode-badge">
          Heatmap Mode · {n}×{d} = {(n * d).toLocaleString()} gates
        </span>
      </h2>
      <div className="heatmap-canvas-container">
        <canvas
          ref={canvasRef}
          style={{ display: "block" }}
        />
        <canvas
          ref={overlayCanvasRef}
          className="heatmap-overlay-canvas"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
        />
        <div className="heatmap-axes">
          <span className="axis-label-x">Layer →</span>
          <span className="axis-label-y">Wire ↓</span>
        </div>
        {/* Color legend */}
        <div className="heatmap-legend">
          <span className="legend-label">−1</span>
          <div className="legend-gradient" />
          <span className="legend-label">+1</span>
        </div>
      </div>
      {hovered && (
        <GateDetailOverlay
          circuit={circuit}
          means={means}
          hoveredWire={hovered.wire}
          hoveredLayer={hovered.layer}
          position={{ x: hovered.x, y: hovered.y }}
        />
      )}
    </div>
  );
}
