/**
 * CoeffHistograms — Per-layer coefficient distribution band charts.
 * Shows 4 separate subplots (c, a, b, p), each as a mean ±σ band chart
 * showing how the distribution of coefficient values evolves across layers.
 * Canvas-rendered for performance. Supports 1- or 2-column layout.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CanvasTooltip from "./CanvasTooltip";
import InfoTip from "./InfoTip";

const COLORS = {
  c: { line: "#8B95A2", fill: "139,149,162" },  // medium gray (was near-white)
  a: { line: "#5B7BA8", fill: "91,123,168" },    // steel blue
  b: { line: "#1E293B", fill: "30,41,59" },      // near-black slate
  p: { line: "#F0524D", fill: "240,82,77" },     // coral
};
const COEFF_KEYS = ["c", "a", "b", "p"];
const COEFF_LABELS = {
  c: "constant bias", a: "first input (x)", b: "second input (y)", p: "interaction (x·y)",
};

export default function CoeffHistograms({ circuit, columns = 1 }) {
  const canvasRef = useRef(null);
  const layoutRef = useRef(null);
  const [hover, setHover] = useState(null);

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

    const cols = Math.min(columns, COEFF_KEYS.length);
    const rows = Math.ceil(COEFF_KEYS.length / cols);
    const COL_GAP = cols > 1 ? 24 : 0;
    const SUBPLOT_H = 120;
    const ROW_GAP = 8;
    const PAD = { top: 16, bottom: 28, left: 44, right: 10 };
    const colW = (W - COL_GAP * (cols - 1)) / cols;
    const totalH = rows * (SUBPLOT_H + ROW_GAP) + PAD.bottom;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${totalH}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, totalH);

    const d = layerStats.length;

    // Store layout for hover
    layoutRef.current = { PAD, cols, rows, COL_GAP, SUBPLOT_H, ROW_GAP, colW, d, W };

    for (let ki = 0; ki < COEFF_KEYS.length; ki++) {
      const key = COEFF_KEYS[ki];
      const color = COLORS[key];
      const row = Math.floor(ki / cols);
      const col = ki % cols;
      const xOffset = col * (colW + COL_GAP);
      const yOffset = row * (SUBPLOT_H + ROW_GAP);
      const plotW = colW - PAD.left - PAD.right;
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
      const absMax = Math.max(Math.abs(globalMin), Math.abs(globalMax), 0.1);
      const rangeMin = -absMax * 1.1;
      const rangeMax = absMax * 1.1;

      const xScale = (l) => xOffset + PAD.left + (l / Math.max(1, d - 1)) * plotW;
      const yScale = (v) => yOffset + PAD.top + (1 - (v - rangeMin) / (rangeMax - rangeMin)) * plotH;

      // Subplot label
      ctx.fillStyle = color.line;
      ctx.font = "bold 10px 'IBM Plex Mono', monospace";
      ctx.textAlign = "left";
      ctx.fillText(COEFF_LABELS[key], xOffset + PAD.left, yOffset + 11);

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
      ctx.moveTo(xOffset + PAD.left, yScale(0));
      ctx.lineTo(xOffset + PAD.left + plotW, yScale(0));
      ctx.stroke();
      ctx.setLineDash([]);

      // Y-axis labels
      ctx.fillStyle = "#9CA3AF";
      ctx.font = "8px 'IBM Plex Mono', monospace";
      ctx.textAlign = "right";
      const yTicks = [rangeMax, 0, rangeMin];
      for (const v of yTicks) {
        const label = v === 0 ? "0" : v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1);
        ctx.fillText(label, xOffset + PAD.left - 4, yScale(v) + 3);
      }

      // X-axis labels per subplot
      ctx.fillStyle = "#9CA3AF";
      ctx.font = "9px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      const labelStep = Math.max(1, Math.floor(d / 10));
      const plotBottom = yOffset + SUBPLOT_H - 4;
      for (let l = 0; l < d; l += labelStep) {
        ctx.fillText(`${l}`, xScale(l), plotBottom + 12);
      }
      ctx.fillText("Layer", xOffset + PAD.left + plotW / 2, plotBottom + 24);
    }
  }, [layerStats, columns]);

  const handleMouseMove = useCallback((e) => {
    if (!layoutRef.current || layerStats.length === 0) return;
    const { PAD, cols, SUBPLOT_H, ROW_GAP, COL_GAP, colW, d } = layoutRef.current;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Determine which subplot row/col we're in
    const row = Math.floor(my / (SUBPLOT_H + ROW_GAP));
    const col = Math.floor(mx / (colW + COL_GAP));
    const ki = row * cols + col;
    if (ki < 0 || ki >= COEFF_KEYS.length) { setHover(null); return; }

    // Determine layer within subplot
    const xOffset = col * (colW + COL_GAP);
    const plotW = colW - PAD.left - PAD.right;
    const localX = mx - xOffset - PAD.left;
    const frac = localX / plotW;
    const layer = Math.round(frac * (d - 1));

    if (layer >= 0 && layer < d) {
      setHover({ layer, pageX: e.pageX, pageY: e.pageY });
    } else {
      setHover(null);
    }
  }, [layerStats]);

  const handleMouseLeave = useCallback(() => setHover(null), []);

  if (!circuit) return null;

  const hStats = hover ? layerStats[hover.layer] : null;

  return (
    <div className="panel">
      <h2>
        Coefficient Distributions
        <InfoTip>
          <span className="tip-title">Coefficient Distributions</span>
          <p className="tip-desc">
            Four band charts — one per coefficient type — showing how gate coefficient distributions evolve across layers.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key"><span className="tip-mono">c</span></span><span className="tip-kv-val">Constant bias</span></div>
          <div className="tip-kv"><span className="tip-kv-key"><span className="tip-mono">a</span></span><span className="tip-kv-val">First input (x)</span></div>
          <div className="tip-kv"><span className="tip-kv-key"><span className="tip-mono">b</span></span><span className="tip-kv-val">Second input (y)</span></div>
          <div className="tip-kv"><span className="tip-kv-key"><span className="tip-mono">p</span></span><span className="tip-kv-val">Interaction (x·y)</span></div>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Solid line</span><span className="tip-kv-val">Per-layer mean</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Bands</span><span className="tip-kv-val">±σ and min–max range</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            Stable distributions suggest uniform structure; diverging bands indicate layer-dependent patterns.
          </p>
        </InfoTip>
      </h2>
      <div style={{ width: "100%", overflowX: "auto" }}>
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ cursor: "crosshair" }}
        />
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
      <CanvasTooltip visible={!!hover} pageX={hover?.pageX} pageY={hover?.pageY}>
        {hStats && (
          <>
            <div className="canvas-tip-header">
              Layer <span className="layer-num">{hover.layer}</span>
            </div>
            <div className="canvas-tip-rows">
              {COEFF_KEYS.map(k => (
                <div key={k}>
                  <div className="canvas-tip-row">
                    <span className="canvas-tip-label">
                      <span className="canvas-tip-swatch" style={{ background: COLORS[k].line }} />
                      {k} — {COEFF_LABELS[k]}  mean
                    </span>
                    <span className="canvas-tip-value">{hStats[k].mean.toFixed(4)}</span>
                  </div>
                  <div className="canvas-tip-row">
                    <span className="canvas-tip-label" style={{ paddingLeft: 14 }}>
                      σ
                    </span>
                    <span className="canvas-tip-value">
                      ±{hStats[k].std.toFixed(4)}
                      <span className="canvas-tip-sub">
                        [{hStats[k].min.toFixed(2)}, {hStats[k].max.toFixed(2)}]
                      </span>
                    </span>
                  </div>
                  {k !== "p" && <div className="canvas-tip-divider" />}
                </div>
              ))}
            </div>
          </>
        )}
      </CanvasTooltip>
    </div>
  );
}
