/**
 * SignalHeatmap — Neuron means heatmap using canvas rendering.
 * Uses meanToColor for exact palette matching (dark slate → white → coral).
 * Orientation: X-axis = Layer, Y-axis = Neuron (matches network layout).
 */
import { useEffect, useRef } from "react";
import InfoTip from "./InfoTip";

// Activation magnitude color: 0 = dark slate, high = coral
function meanToColor(v) {
  const t = Math.max(0, Math.min(1, v));
  const r = Math.round(51 + 204 * t);
  const g = Math.round(65 - 65 * t);
  const b = Math.round(85 - 85 * t);
  return `rgb(${r},${g},${b})`;
}

export default function SignalHeatmap({ means, width: n, depth: d, source }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!means || means.length === 0 || !canvasRef.current) return;

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

    // Draw cells: X = layer, Y = neuron
    const gapW = cellW > 3 ? 1 : 0;
    const gapH = cellH > 3 ? 1 : 0;
    for (let l = 0; l < d && l < means.length; l++) {
      for (let w = 0; w < n; w++) {
        const v = means[l]?.[w] ?? 0;
        ctx.fillStyle = meanToColor(v) || "#FFFFFF";
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

    // Y-axis labels: Neuron
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
    ctx.fillText("Neuron", 0, 0);
    ctx.restore();
  }, [means, n, d]);

  if (!means || means.length === 0) return null;

  return (
    <div className="panel">
      <h2>
        Neuron Means Heatmap
        <InfoTip>
          <span className="tip-title">Neuron Means Heatmap</span>
          <p className="tip-desc">
            Color-coded grid of <span className="tip-mono">E[neuron]</span> values for every neuron at every layer.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Dark slate</span><span className="tip-kv-val">E[neuron] near 0</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Coral</span><span className="tip-kv-val">E[neuron] high</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            Patterns here reveal structural biases in the network weights.
          </p>
        </InfoTip>
        {source && <span className="source-badge">{source}</span>}
      </h2>
      <div style={{ width: '100%', overflowX: 'hidden' }}>
        <canvas ref={canvasRef} />
      </div>
      <div className="heatmap-legend">
        <span className="legend-label">−1</span>
        <div className="legend-gradient" />
        <span className="legend-label">+1</span>
      </div>
    </div>
  );
}
