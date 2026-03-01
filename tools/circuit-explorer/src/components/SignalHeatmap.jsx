/**
 * SignalHeatmap — Wire means heatmap using canvas rendering.
 * Uses meanToColor for exact palette matching (dark slate → white → coral).
 */
import { useEffect, useRef } from "react";
import { meanToColor } from "./gateShapes";

export default function SignalHeatmap({ means, width: n, depth: d, source }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!means || means.length === 0 || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const containerW = container.offsetWidth || 500;

    // Cell sizing — fit to container
    const LABEL_PAD_Y = 28; // space for wire labels (x-axis)
    const LABEL_PAD_X = 36; // space for layer labels (y-axis)
    const cellW = Math.max(4, Math.floor((containerW - LABEL_PAD_X - 20) / n));
    const cellH = Math.max(4, Math.min(20, Math.floor(280 / d)));
    const chartW = cellW * n;
    const chartH = cellH * d;

    const totalW = chartW + LABEL_PAD_X + 10;
    const totalH = chartH + LABEL_PAD_Y + 10;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = totalW * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = `${totalW}px`;
    canvas.style.height = `${totalH}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, totalW, totalH);

    // Draw heatmap cells
    for (let l = 0; l < d && l < means.length; l++) {
      for (let w = 0; w < n; w++) {
        const v = means[l]?.[w] ?? 0;
        ctx.fillStyle = meanToColor(v) || "#FFFFFF";
        ctx.fillRect(LABEL_PAD_X + w * cellW, l * cellH, cellW - 1, cellH - 1);
      }
    }

    // Draw axis labels
    ctx.fillStyle = "#64748B";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";

    // X-axis: wire indices
    const wireLabelStep = n > 16 ? Math.ceil(n / 8) : 1;
    for (let w = 0; w < n; w++) {
      if (w % wireLabelStep === 0) {
        ctx.fillText(`${w}`, LABEL_PAD_X + w * cellW + cellW / 2, chartH + 14);
      }
    }

    // Y-axis: layer indices
    const layerLabelStep = d > 16 ? Math.ceil(d / 8) : 1;
    ctx.textAlign = "right";
    for (let l = 0; l < d; l++) {
      if (l % layerLabelStep === 0) {
        ctx.fillText(`${l}`, LABEL_PAD_X - 4, l * cellH + cellH / 2 + 3);
      }
    }

    // Axis titles
    ctx.fillStyle = "#94A3B8";
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Wire", LABEL_PAD_X + chartW / 2, chartH + 26);
    ctx.save();
    ctx.translate(10, chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Layer", 0, 0);
    ctx.restore();
  }, [means, n, d]);

  if (!means || means.length === 0) return null;

  return (
    <div className="panel">
      <h2>
        Wire Means Heatmap
        {source && <span className="source-badge">{source}</span>}
      </h2>
      <div style={{ width: '100%', overflowX: 'auto' }}>
        <canvas ref={canvasRef} />
      </div>
      <div className="heatmap-legend">
        <span className="legend-item" style={{ color: "#334155" }}>◆ −1</span>
        <span className="legend-item" style={{ color: "#9CA3AF" }}>◆ 0</span>
        <span className="legend-item" style={{ color: "#F0524D" }}>◆ +1</span>
      </div>
    </div>
  );
}
