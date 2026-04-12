/**
 * NetworkHeatmap — Canvas-rendered neurons × layers heatmap
 * for large networks (n×d > 4096).
 * Shows neuron means as a color grid, with hover triggering a rich tooltip
 * that displays activation, incoming weights, and outgoing weights.
 *
 * Performance notes:
 * - Uses a main canvas for the heatmap grid (redrawn only when data changes)
 * - Uses a separate overlay canvas for crosshair (redrawn on mousemove — cheap)
 * - Debounces hover: RAF-gated, only updates when the hovered cell changes
 * - Resolution cap: sub-pixel cells are rendered at reduced resolution
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { perfEnd, perfStart } from "../perf";

/* ── Weight color helpers (match NetworkGraph palette) ── */
function weightColorRGB(w) {
  if (Math.abs(w) < 0.001) return [170, 172, 173];
  if (w < 0) {
    const t = Math.min(1, Math.abs(w) * 2);
    return [
      Math.round(170 + (51 - 170) * t),
      Math.round(172 + (65 - 172) * t),
      Math.round(173 + (85 - 173) * t),
    ];
  }
  const t = Math.min(1, w * 2);
  return [
    Math.round(170 + (240 - 170) * t),
    Math.round(172 + (82 - 172) * t),
    Math.round(173 + (77 - 173) * t),
  ];
}

function weightColor(w) {
  const [r, g, b] = weightColorRGB(w);
  return `rgb(${r},${g},${b})`;
}

/** Extract top-N connections by |weight|, sorted descending */
function topConnections(weights, width, neuronIdx, direction, layerIdx, N = 5) {
  if (!weights || layerIdx < 0 || layerIdx >= weights.length) return [];
  const W = weights[layerIdx];
  if (!W) return [];
  const conns = [];
  for (let k = 0; k < width; k++) {
    const idx = direction === "in" ? k * width + neuronIdx : neuronIdx * width + k;
    const w = W[idx];
    if (w !== undefined) conns.push({ neuron: k, weight: w });
  }
  conns.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  return conns.slice(0, N);
}

/* ── Vertical weight strip rendered as inline SVG ── */
const STRIP_W = 24;          // vertical column width
const MAX_STRIP_H = 180;     // cap height so tooltip stays manageable

function WeightStrip({ weights, width, neuronIdx, direction, layerIdx }) {
  if (!weights || layerIdx < 0 || layerIdx >= weights.length) return null;
  const W = weights[layerIdx];
  if (!W) return null;

  const stripH = Math.min(MAX_STRIP_H, Math.max(60, width * 2));
  const cellH = stripH / width;

  const rects = [];
  for (let k = 0; k < width; k++) {
    const idx = direction === "in" ? k * width + neuronIdx : neuronIdx * width + k;
    const w = W[idx];
    rects.push(
      <rect
        key={k}
        x={0}
        y={k * cellH}
        width={STRIP_W}
        height={Math.max(cellH, 0.5)}
        fill={weightColor(w)}
      />
    );
  }

  return (
    <svg width={STRIP_W} height={stripH} style={{ display: "block", borderRadius: 2, border: "1px solid var(--gray-100)", flexShrink: 0 }}>
      {rects}
      {/* Highlight the hovered neuron's own row */}
      <rect
        x={0}
        y={neuronIdx * cellH}
        width={STRIP_W}
        height={Math.max(cellH, 1)}
        fill="none"
        stroke="#fff"
        strokeWidth={1.5}
      />
    </svg>
  );
}

/* ── Connection list: top N by |weight|, rendered vertically ── */
function ConnectionList({ conns, align = "left" }) {
  if (!conns || conns.length === 0) return null;
  const isRight = align === "right";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 1, alignItems: isRight ? "flex-end" : "flex-start" }}>
      {conns.map(({ neuron, weight }) => (
        <span key={neuron} style={{
          fontFamily: "var(--font-mono)", fontSize: 10,
          display: "inline-flex", alignItems: "center", gap: 3,
          flexDirection: isRight ? "row-reverse" : "row",
        }}>
          <span style={{
            display: "inline-block", width: 6, height: 6, borderRadius: 1,
            background: weightColor(weight), flexShrink: 0,
          }} />
          <span style={{ color: "var(--gray-500)" }}>n{neuron}</span>
          <span style={{ color: "var(--gray-800)", fontWeight: 600 }}>
            {weight >= 0 ? "+" : ""}{weight.toFixed(2)}
          </span>
        </span>
      ))}
    </div>
  );
}

export default function NetworkHeatmap({ mlp, means, activeLayer, onLayerClick }) {
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const containerRef = useRef(null);
  const [hovered, setHovered] = useState(null); // { neuron, layer, pageX, pageY, value }
  const [dims, setDims] = useState({ cellW: 0, cellH: 0 });
  const lastCellRef = useRef(null);
  const rafRef = useRef(null);

  const n = mlp.width;
  const d = mlp.depth;
  const { weights } = mlp;

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
            pageX: e.pageX,
            pageY: e.pageY,
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

  // Compute connection data for the hovered neuron (memoized)
  const connectionData = useMemo(() => {
    if (!hovered || !weights) return null;
    const { neuron, layer } = hovered;
    // Incoming: weights[layer] maps from previous layer → this layer
    // weights[l][i * width + neuron] = weight from neuron i to this neuron
    const hasIncoming = layer >= 0 && layer < weights.length;
    const inConns = hasIncoming
      ? topConnections(weights, n, neuron, "in", layer, 5)
      : [];
    // Outgoing: weights[layer+1] maps from this layer → next layer
    // weights[l+1][neuron * width + j] = weight from this neuron to neuron j
    const hasOutgoing = (layer + 1) >= 0 && (layer + 1) < weights.length;
    const outConns = hasOutgoing
      ? topConnections(weights, n, neuron, "out", layer + 1, 5)
      : [];
    return {
      hasIncoming, inConns,
      hasOutgoing, outConns,
      inLayerIdx: layer,
      outLayerIdx: layer + 1,
    };
  }, [hovered, weights, n]);

  // Tooltip positioning (portal to body, viewport-aware)
  const hasConnections = connectionData?.hasIncoming || connectionData?.hasOutgoing;
  const TOOLTIP_W = hasConnections ? 340 : 200;
  const vpW = typeof window !== "undefined" ? window.innerWidth : 1920;
  const vpH = typeof window !== "undefined" ? window.innerHeight : 1080;
  const tooltipLeft = hovered
    ? (hovered.pageX + TOOLTIP_W + 30 > vpW
      ? hovered.pageX - TOOLTIP_W - 24
      : hovered.pageX + 16)
    : 0;
  const tooltipTop = hovered ? Math.max(8, hovered.pageY - 20) : 0;

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
      {hovered && createPortal(
        <div
          className="canvas-data-tooltip heatmap-connection-tooltip"
          style={{ left: tooltipLeft, top: tooltipTop }}
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

          {/* Connection weight strips — symmetric mirror layout */}
          {(connectionData?.hasIncoming || connectionData?.hasOutgoing) && (
            <>
              <div className="canvas-tip-divider" />
              {/* Headers row */}
              <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 10px 2px" }}>
                {connectionData.hasIncoming && (
                  <div className="heatmap-tip-section-label" style={{ padding: 0, textAlign: "center" }}>
                    In
                    <span className="canvas-tip-sub">L{hovered.layer - 1 >= 0 ? hovered.layer - 1 : "in"}</span>
                  </div>
                )}
                {connectionData.hasOutgoing && (
                  <div className="heatmap-tip-section-label" style={{ padding: 0, textAlign: "center" }}>
                    Out
                    <span className="canvas-tip-sub">L{hovered.layer + 1}</span>
                  </div>
                )}
              </div>
              {/* Strips + connection lists: text|strip || strip|text */}
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0 10px 4px" }}>
                {/* Incoming: text on left, strip on right */}
                {connectionData.hasIncoming && (
                  <div style={{ display: "flex", gap: 5, alignItems: "flex-start", flexDirection: "row-reverse" }}>
                    <WeightStrip
                      weights={weights}
                      width={n}
                      neuronIdx={hovered.neuron}
                      direction="in"
                      layerIdx={connectionData.inLayerIdx}
                    />
                    <ConnectionList conns={connectionData.inConns} align="right" />
                  </div>
                )}

                {/* Outgoing: strip on left, text on right */}
                {connectionData.hasOutgoing && (
                  <div style={{ display: "flex", gap: 5, alignItems: "flex-start" }}>
                    <WeightStrip
                      weights={weights}
                      width={n}
                      neuronIdx={hovered.neuron}
                      direction="out"
                      layerIdx={connectionData.outLayerIdx}
                    />
                    <ConnectionList conns={connectionData.outConns} align="left" />
                  </div>
                )}
              </div>

              {/* Explanatory footer */}
              <div className="canvas-tip-divider" />
              <div style={{ padding: "2px 10px 6px", fontFamily: "var(--font-mono)", fontSize: 9, color: "var(--gray-400)", lineHeight: 1.4 }}>
                <span>n<em>k</em> = neuron k · weight value from strongest connections</span>
                <div className="heatmap-tip-legend" style={{ padding: "2px 0 0" }}>
                  <span style={{ color: "#334155" }}>━ neg</span>
                  <span style={{ color: "#AAACAD" }}>━ ~0</span>
                  <span style={{ color: "#F0524D" }}>━ pos</span>
                </div>
              </div>
            </>
          )}
        </div>,
        document.body
      )}
    </div>
  );
}
