/**
 * NetworkHeatmap — Canvas-rendered neurons × layers heatmap
 * for large networks (n×d > 4096).
 * Shows neuron means as a color grid, with hover triggering a simple info tooltip.
 *
 * Performance notes:
 * - Uses a main canvas for the heatmap grid (redrawn only when data changes)
 * - Uses a separate overlay canvas for crosshair (redrawn on mousemove — cheap)
 * - Debounces hover: RAF-gated, only updates when the hovered cell changes
 * - Resolution cap: sub-pixel cells are rendered at reduced resolution
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { perfEnd, perfStart } from "../perf";

export default function NetworkHeatmap({ mlp, means, activeLayer, onLayerClick }) {
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const containerRef = useRef(null);
  const [hovered, setHovered] = useState(null); // { neuron, layer, x, y, value }
  const [dims, setDims] = useState({ cellW: 0, cellH: 0 });
  const lastCellRef = useRef(null);
  const rafRef = useRef(null);

  const n = mlp.width;
  const d = mlp.depth;

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
      const neuron = Math.floor((py / canvasH) * n);
      for (let px = 0; px < canvasW; px++) {
        const layer = Math.floor((px / canvasW) * d);
        const idx = (py * canvasW + px) * 4;
        const mean = means && means[layer] ? means[layer][neuron] : null;

        if (mean !== null && mean !== undefined) {
          // Activation-magnitude color scale: 0 = dark/gray, high = bright coral
          const t = Math.max(0, Math.min(1, mean));
          pixels[idx]     = (51 + (204 * t)) | 0;
          pixels[idx + 1] = (65  - (65  * t)) | 0;
          pixels[idx + 2] = (85  - (85  * t)) | 0;
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
      for (let w = 0; w <= n; w++) { // neuron rows
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
  }, [mlp, means, n, d]);

  // Draw crosshair + activeLayer column on overlay canvas
  const drawCrosshair = useCallback((layer, neuron) => {
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

    if (layer === null || neuron === null) return;

    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 1;

    // Vertical line (layer column)
    const lx = layer * dims.cellW + dims.cellW / 2;
    ctx.beginPath();
    ctx.moveTo(lx, 0);
    ctx.lineTo(lx, dims.height);
    ctx.stroke();

    // Horizontal line (neuron row)
    const wy = neuron * dims.cellH + dims.cellH / 2;
    ctx.beginPath();
    ctx.moveTo(0, wy);
    ctx.lineTo(dims.width, wy);
    ctx.stroke();

    // Cell highlight border
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(
      layer * dims.cellW,
      neuron * dims.cellH,
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
        const neuron = Math.floor(my / dims.cellH);
        const cellKey = `${layer},${neuron}`;

        if (cellKey === lastCellRef.current) return; // same cell, skip
        lastCellRef.current = cellKey;

        if (layer >= 0 && layer < d && neuron >= 0 && neuron < n) {
          drawCrosshair(layer, neuron);
          const value = means && means[layer] ? (means[layer][neuron] ?? null) : null;
          setHovered({
            neuron,
            layer,
            value,
            x: e.clientX - containerRef.current.getBoundingClientRect().left,
            y: e.clientY - containerRef.current.getBoundingClientRect().top,
          });
        } else {
          drawCrosshair(null, null);
          setHovered(null);
        }
      });
    },
    [dims, n, d, drawCrosshair, means]
  );

  const handleMouseLeave = useCallback(() => {
    lastCellRef.current = null;
    drawCrosshair(null, null);
    setHovered(null);
  }, [drawCrosshair]);

  return (
    <div className="panel network-heatmap" ref={containerRef} style={{ position: "relative" }}>
      <h2>
        Network Structure
        <span className="mode-badge">
          Heatmap Mode · {n}×{d} = {(n * d).toLocaleString()} neurons
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
          <span className="axis-label-y">Neuron ↓</span>
        </div>
        {/* Color legend */}
        <div className="heatmap-legend">
          <span className="legend-label">0</span>
          <div className="legend-gradient" />
          <span className="legend-label">high</span>
        </div>
      </div>
      {hovered && (
        <div
          className="canvas-data-tooltip"
          style={{
            position: "absolute",
            left: hovered.x + 16,
            top: hovered.y - 20,
            pointerEvents: "none",
          }}
        >
          <div className="canvas-tip-header">
            Neuron <span className="layer-num">{hovered.neuron}</span>
            {" · "}
            Layer <span className="layer-num">{hovered.layer}</span>
          </div>
          <div className="canvas-tip-rows">
            <div className="canvas-tip-row">
              <span className="canvas-tip-label">Activation</span>
              <span className="canvas-tip-value">
                {hovered.value !== null ? hovered.value.toFixed(4) : "—"}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
