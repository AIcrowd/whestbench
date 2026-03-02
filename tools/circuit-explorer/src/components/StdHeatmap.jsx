/**
 * StdHeatmap — Wire × Layer standard deviation heatmap.
 * Shows where activations are most variable (input-dependent).
 * Low σ = dark (predictable), High σ = bright coral (variable).
 * Orientation: X-axis = Layer, Y-axis = Wire (matches circuit layout).
 */
import { useEffect, useRef } from "react";

function stdToColor(std, maxStd) {
  const t = Math.min(1, std / Math.max(0.01, maxStd));
  const mapped = t * 2 - 1;
  if (mapped < 0) {
    const s = 1 + mapped;
    const r = Math.round(51 + (255 - 51) * s);
    const g = Math.round(65 + (255 - 65) * s);
    const b = Math.round(85 + (255 - 85) * s);
    return `rgb(${r},${g},${b})`;
  } else {
    const r = Math.round(255 - (255 - 240) * mapped);
    const g = Math.round(255 - (255 - 82) * mapped);
    const b = Math.round(255 - (255 - 77) * mapped);
    return `rgb(${r},${g},${b})`;
  }
}

export default function StdHeatmap({ stds, width: n, depth: d }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!stds || stds.length === 0 || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const containerW = container.offsetWidth || 500;

    const LABEL_PAD_Y = 28;
    const LABEL_PAD_X = 36;
    const availW = containerW - LABEL_PAD_X - 10;

    // Transposed: X = layers (d columns), Y = wires (n rows)
    const cellW = availW / d;
    const MAX_CHART_H = 150;
    const cellH = Math.min(Math.max(1, Math.floor(MAX_CHART_H / n)), 12);
    const chartW = cellW * d;
    const chartH = cellH * n;

    const totalW = LABEL_PAD_X + chartW + 10;
    const totalH = chartH + LABEL_PAD_Y + 10;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = totalW * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = `${totalW}px`;
    canvas.style.height = `${totalH}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, totalW, totalH);

    // Find global max std
    let maxStd = 0;
    for (let l = 0; l < d && l < stds.length; l++) {
      for (let w = 0; w < n; w++) {
        if (stds[l][w] > maxStd) maxStd = stds[l][w];
      }
    }

    // Draw cells: X = layer, Y = wire
    const gapW = cellW > 3 ? 1 : 0;
    const gapH = cellH > 3 ? 1 : 0;
    for (let l = 0; l < d && l < stds.length; l++) {
      for (let w = 0; w < n; w++) {
        ctx.fillStyle = stdToColor(stds[l][w], maxStd);
        ctx.fillRect(
          LABEL_PAD_X + l * cellW,
          w * cellH,
          Math.max(1, cellW - gapW),
          Math.max(1, cellH - gapH)
        );
      }
    }

    // X-axis labels: Layer
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    const layerStep = d > 16 ? Math.ceil(d / 8) : 1;
    for (let l = 0; l < d; l++) {
      if (l % layerStep === 0) {
        ctx.fillText(`${l}`, LABEL_PAD_X + l * cellW + cellW / 2, chartH + 14);
      }
    }

    // Y-axis labels: Wire
    const wireStep = n > 16 ? Math.ceil(n / 8) : 1;
    ctx.textAlign = "right";
    for (let w = 0; w < n; w++) {
      if (w % wireStep === 0) {
        ctx.fillText(`${w}`, LABEL_PAD_X - 4, w * cellH + cellH / 2 + 3);
      }
    }

    // Axis titles
    ctx.fillStyle = "#94A3B8";
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Layer", LABEL_PAD_X + chartW / 2, chartH + 26);
    ctx.save();
    ctx.translate(10, chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Wire", 0, 0);
    ctx.restore();
  }, [stds, n, d]);

  if (!stds || stds.length === 0) return null;

  return (
    <div className="panel">
      <h2>Signal Variability (σ)</h2>
      <div style={{ width: "100%", overflowX: "hidden" }}>
        <canvas ref={canvasRef} />
      </div>
      <div className="heatmap-legend">
        <span className="legend-label">Low σ</span>
        <div className="legend-gradient" />
        <span className="legend-label">High σ</span>
      </div>
      <p className="panel-desc">
        Wires with high σ are strongly input-dependent.
        Wires with low σ produce nearly constant output regardless of inputs.
      </p>
    </div>
  );
}
