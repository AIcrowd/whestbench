/**
 * CircuitHeatmap — Canvas-rendered wires × layers heatmap
 * for large circuits (n×d > 4096).
 * Shows wire means as a color grid, with hover triggering a detail overlay.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import GateDetailOverlay from "./GateDetailOverlay";
import { meanToColor } from "./gateShapes";

export default function CircuitHeatmap({ circuit, means }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [hovered, setHovered] = useState(null); // { wire, layer, x, y }
  const [dims, setDims] = useState({ cellW: 0, cellH: 0 });

  const n = circuit.n;
  const d = circuit.d;

  // Render heatmap to canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const width = rect.width;
    const height = Math.max(300, Math.min(600, n * 3)); // adaptive height

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const cellW = width / d;
    const cellH = height / n;
    setDims({ cellW, cellH, width, height });

    // Clear
    ctx.fillStyle = "#F9FAFB";
    ctx.fillRect(0, 0, width, height);

    // Draw cells
    for (let l = 0; l < d; l++) {
      for (let w = 0; w < n; w++) {
        const mean = means && means[l] ? means[l][w] : null;
        ctx.fillStyle = mean !== null ? meanToColor(mean) : "#E5E7EB";
        ctx.fillRect(l * cellW, w * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // Grid lines (only if cells are big enough)
    if (cellW > 3 && cellH > 3) {
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
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    // Layer labels at bottom
    const labelStep = Math.max(1, Math.floor(d / 10));
    for (let l = 0; l < d; l += labelStep) {
      ctx.fillText(`${l}`, l * cellW + cellW / 2, height - 2);
    }
  }, [circuit, means, n, d]);

  // Handle mouse move to detect hovered cell
  const handleMouseMove = useCallback(
    (e) => {
      if (!dims.cellW) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const layer = Math.floor(mx / dims.cellW);
      const wire = Math.floor(my / dims.cellH);

      if (layer >= 0 && layer < d && wire >= 0 && wire < n) {
        setHovered({
          wire,
          layer,
          x: e.clientX - containerRef.current.getBoundingClientRect().left,
          y: e.clientY - containerRef.current.getBoundingClientRect().top,
        });
      } else {
        setHovered(null);
      }
    },
    [dims, n, d]
  );

  return (
    <div className="panel circuit-heatmap" ref={containerRef}>
      <h2>
        Circuit Structure
        <span className="mode-badge">
          Heatmap Mode · {n}×{d} = {(n * d).toLocaleString()} gates
        </span>
      </h2>
      <div style={{ position: "relative" }}>
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHovered(null)}
          style={{ cursor: "crosshair", display: "block", width: "100%" }}
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
