/**
 * CoeffHistograms — Per-layer coefficient distribution band charts.
 * Shows 4 separate subplots (c, a, b, p), each as a mean ±σ band chart
 * showing how the distribution of coefficient values evolves across layers.
 * Canvas-rendered for performance.
 */
import { useEffect, useMemo, useRef } from "react";

const COLORS = {
  c: { line: "#8B95A2", fill: "139,149,162" },  // medium gray (was near-white)
  a: { line: "#5B7BA8", fill: "91,123,168" },    // steel blue
  b: { line: "#1E293B", fill: "30,41,59" },      // near-black slate
  p: { line: "#F0524D", fill: "240,82,77" },     // coral
};
const COEFF_KEYS = ["c", "a", "b", "p"];
const COEFF_LABELS = {
  c: "bias (c)", a: "first (a)", b: "second (b)", p: "product (p)",
};

export default function CoeffHistograms({ circuit }) {
  const canvasRef = useRef(null);

  const layerStats = useMemo(() => {
    if (!circuit) return [];
    const { n, d, gates } = circuit;
    const result = [];
    for (let l = 0; l < d; l++) {
      const layer = gates[l];
      const coeffValues = {
        c: layer.const, a: layer.firstCoeff, b: layer.secondCoeff, p: layer.productCoeff,
      };
      const stats = {};
      for (const key of COEFF_KEYS) {
        const vals = coeffValues[key];
        let sum = 0, sumSq = 0, mn = Infinity, mx = -Infinity;
        for (let i = 0; i < n; i++) {
          const v = vals[i];
          sum += v;
          sumSq += v * v;
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
        const mean = sum / n;
        const variance = sumSq / n - mean * mean;
        const std = Math.sqrt(Math.max(0, variance));
        stats[key] = { mean, std, min: mn, max: mx };
      }
      result.push(stats);
    }
    return result;
  }, [circuit]);

  useEffect(() => {
    if (!canvasRef.current || layerStats.length === 0) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const W = container.offsetWidth || 600;

    const SUBPLOT_H = 120;
    const GAP = 8;
    const PAD = { top: 16, bottom: 28, left: 44, right: 10 };
    const totalH = COEFF_KEYS.length * (SUBPLOT_H + GAP) + PAD.bottom;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${totalH}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, totalH);

    const d = layerStats.length;
    const plotW = W - PAD.left - PAD.right;

    for (let ki = 0; ki < COEFF_KEYS.length; ki++) {
      const key = COEFF_KEYS[ki];
      const color = COLORS[key];
      const yOffset = ki * (SUBPLOT_H + GAP);
      const plotH = SUBPLOT_H - PAD.top - 4;

      // Find global range for this coefficient across all layers
      let globalMin = Infinity, globalMax = -Infinity;
      for (let l = 0; l < d; l++) {
        const s = layerStats[l][key];
        const lo = Math.min(s.min, s.mean - 2 * s.std);
        const hi = Math.max(s.max, s.mean + 2 * s.std);
        if (lo < globalMin) globalMin = lo;
        if (hi > globalMax) globalMax = hi;
      }
      // Ensure symmetric range around 0 for visual clarity
      const absMax = Math.max(Math.abs(globalMin), Math.abs(globalMax), 0.1);
      const rangeMin = -absMax * 1.1;
      const rangeMax = absMax * 1.1;

      const xScale = (l) => PAD.left + (l / Math.max(1, d - 1)) * plotW;
      const yScale = (v) => yOffset + PAD.top + (1 - (v - rangeMin) / (rangeMax - rangeMin)) * plotH;

      // Subplot label
      ctx.fillStyle = color.line;
      ctx.font = "bold 10px 'IBM Plex Mono', monospace";
      ctx.textAlign = "left";
      ctx.fillText(COEFF_LABELS[key], PAD.left, yOffset + 11);

      // Draw bands: min/max → ±2σ → ±σ
      const bands = [
        { getY: (s) => [s.min, s.max], alpha: 0.08 },
        { getY: (s) => [s.mean - 2 * s.std, s.mean + 2 * s.std], alpha: 0.12 },
        { getY: (s) => [s.mean - s.std, s.mean + s.std], alpha: 0.22 },
      ];

      for (const band of bands) {
        ctx.beginPath();
        for (let l = 0; l < d; l++) {
          const [lo] = band.getY(layerStats[l][key]);
          const x = xScale(l);
          l === 0 ? ctx.moveTo(x, yScale(lo)) : ctx.lineTo(x, yScale(lo));
        }
        for (let l = d - 1; l >= 0; l--) {
          const [, hi] = band.getY(layerStats[l][key]);
          ctx.lineTo(xScale(l), yScale(hi));
        }
        ctx.closePath();
        ctx.fillStyle = `rgba(${color.fill},${band.alpha})`;
        ctx.fill();
      }

      // Mean line
      ctx.beginPath();
      ctx.strokeStyle = color.line;
      ctx.lineWidth = 2;
      for (let l = 0; l < d; l++) {
        const x = xScale(l);
        const y = yScale(layerStats[l][key].mean);
        l === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Zero line
      ctx.beginPath();
      ctx.strokeStyle = "rgba(156,163,175,0.3)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.moveTo(PAD.left, yScale(0));
      ctx.lineTo(PAD.left + plotW, yScale(0));
      ctx.stroke();
      ctx.setLineDash([]);

      // Y-axis labels
      ctx.fillStyle = "#9CA3AF";
      ctx.font = "8px 'IBM Plex Mono', monospace";
      ctx.textAlign = "right";
      const yTicks = [rangeMax, 0, rangeMin];
      for (const v of yTicks) {
        const label = v === 0 ? "0" : v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1);
        ctx.fillText(label, PAD.left - 4, yScale(v) + 3);
      }
    }

    // X-axis labels (shared at bottom)
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    const labelStep = Math.max(1, Math.floor(d / 10));
    const lastYOffset = (COEFF_KEYS.length - 1) * (SUBPLOT_H + GAP);
    const lastPlotBottom = lastYOffset + SUBPLOT_H;
    for (let l = 0; l < d; l += labelStep) {
      const x = PAD.left + (l / Math.max(1, d - 1)) * plotW;
      ctx.fillText(`${l}`, x, lastPlotBottom + 12);
    }
    ctx.fillText("Layer", PAD.left + plotW / 2, lastPlotBottom + 24);
  }, [layerStats]);

  if (!circuit) return null;

  return (
    <div className="panel">
      <h2>Coefficient Distributions</h2>
      <div style={{ width: "100%", overflowX: "auto" }}>
        <canvas ref={canvasRef} />
      </div>
      <div className="formula-legend" style={{ marginTop: 4 }}>
        {COEFF_KEYS.map(k => (
          <span key={k} style={{ color: COLORS[k].line }}>
            ━ <strong>{k}</strong> {COEFF_LABELS[k]}
          </span>
        ))}
      </div>
      <p className="panel-desc">
        Per-layer coefficient distributions showing mean ±σ bands.
        Stable distributions indicate uniform gate structure across layers.
      </p>
    </div>
  );
}
