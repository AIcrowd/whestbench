/**
 * ActivationRibbon — Per-layer activation distribution bands.
 * Shows mean ±σ, ±2σ, and min/max as nested colored bands.
 * Canvas-rendered for performance.
 */
import { useEffect, useRef } from "react";

export default function ActivationRibbon({ means, stds, mins, maxs, depth: d, width: n }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!means || !stds || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const W = container.offsetWidth || 600;
    const H = 200;
    const PAD = { top: 10, bottom: 28, left: 44, right: 10 };
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const layers = Math.min(d, means.length);
    const xScale = (l) => PAD.left + (l / Math.max(1, layers - 1)) * plotW;
    const yScale = (v) => PAD.top + (1 - (v + 1) / 2) * plotH; // [-1,1] → [plotH, 0]

    // Compute per-layer aggregate stats
    const agg = [];
    for (let l = 0; l < layers; l++) {
      let sumMean = 0, sumStd = 0, mn = Infinity, mx = -Infinity;
      for (let w = 0; w < n && w < means[l].length; w++) {
        sumMean += means[l][w];
        sumStd += stds[l][w];
        const lo = mins ? mins[l][w] : means[l][w] - stds[l][w];
        const hi = maxs ? maxs[l][w] : means[l][w] + stds[l][w];
        if (lo < mn) mn = lo;
        if (hi > mx) mx = hi;
      }
      const avgMean = sumMean / n;
      const avgStd = sumStd / n;
      agg.push({ mean: avgMean, std: avgStd, min: mn, max: mx });
    }

    // Draw bands: min/max → ±2σ → ±σ → mean line
    const bands = [
      { getY: (a) => [a.min, a.max], color: "rgba(240,82,77,0.08)" },
      { getY: (a) => [a.mean - 2 * a.std, a.mean + 2 * a.std], color: "rgba(240,82,77,0.12)" },
      { getY: (a) => [a.mean - a.std, a.mean + a.std], color: "rgba(240,82,77,0.22)" },
    ];

    for (const band of bands) {
      ctx.beginPath();
      for (let l = 0; l < layers; l++) {
        const [lo] = band.getY(agg[l]);
        const x = xScale(l);
        l === 0 ? ctx.moveTo(x, yScale(Math.max(-1, lo))) : ctx.lineTo(x, yScale(Math.max(-1, lo)));
      }
      for (let l = layers - 1; l >= 0; l--) {
        const [, hi] = band.getY(agg[l]);
        ctx.lineTo(xScale(l), yScale(Math.min(1, hi)));
      }
      ctx.closePath();
      ctx.fillStyle = band.color;
      ctx.fill();
    }

    // Mean line
    ctx.beginPath();
    ctx.strokeStyle = "#F0524D";
    ctx.lineWidth = 2;
    for (let l = 0; l < layers; l++) {
      const x = xScale(l);
      const y = yScale(agg[l].mean);
      l === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Zero line
    ctx.beginPath();
    ctx.strokeStyle = "rgba(156,163,175,0.4)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.moveTo(PAD.left, yScale(0));
    ctx.lineTo(PAD.left + plotW, yScale(0));
    ctx.stroke();
    ctx.setLineDash([]);

    // Axes
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    const labelStep = Math.max(1, Math.floor(layers / 10));
    for (let l = 0; l < layers; l += labelStep) {
      ctx.fillText(`${l}`, xScale(l), H - 4);
    }
    ctx.fillText("Layer", PAD.left + plotW / 2, H - 14);

    ctx.textAlign = "right";
    ctx.fillText("+1", PAD.left - 4, yScale(1) + 3);
    ctx.fillText("0", PAD.left - 4, yScale(0) + 3);
    ctx.fillText("−1", PAD.left - 4, yScale(-1) + 3);
  }, [means, stds, mins, maxs, d, n]);

  if (!means || !stds) return null;

  return (
    <div className="panel">
      <h2>Activation Distribution</h2>
      <div style={{ width: "100%", overflowX: "auto" }}>
        <canvas ref={canvasRef} />
      </div>
      <div className="formula-legend" style={{ marginTop: 4 }}>
        <span style={{ color: "#F0524D" }}>━ mean</span>
        <span style={{ color: "rgba(240,82,77,0.5)" }}>░ ±σ</span>
        <span style={{ color: "rgba(240,82,77,0.3)" }}>░ ±2σ</span>
        <span style={{ color: "rgba(240,82,77,0.15)" }}>░ min–max</span>
      </div>
    </div>
  );
}
