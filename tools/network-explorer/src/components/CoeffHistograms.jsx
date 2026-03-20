/**
 * CoeffHistograms — Per-layer weight value distribution chart.
 * Shows a mean ±σ band chart of weight values across layers.
 * Canvas-rendered for performance. Panel is collapsible.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import CanvasTooltip from "./CanvasTooltip";
import InfoTip from "./InfoTip";

const LINE_COLOR = "#5B7BA8";
const FILL_RGB = "91,123,168";

export default function CoeffHistograms({ mlp }) {
  const canvasRef = useRef(null);
  const layoutRef = useRef(null);
  const layerStatsRef = useRef([]);
  const [hover, setHover] = useState(null);
  const [collapsed, setCollapsed] = useState(false);

  // Compute per-layer weight stats from mlp.weights
  const layerStats = (() => {
    if (!mlp || !mlp.weights) return [];
    const { depth, width, weights } = mlp;
    const result = [];
    for (let l = 0; l < depth; l++) {
      // weights: Array<Float32Array>, one per layer, each length width×width
      const W = weights[l];
      let sum = 0, sumSq = 0, mn = Infinity, mx = -Infinity;
      const count = W.length;
      for (let i = 0; i < count; i++) {
        const v = W[i];
        sum += v;
        sumSq += v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      const mean = sum / count;
      const variance = sumSq / count - mean * mean;
      const std = Math.sqrt(Math.max(0, variance));
      result.push({ mean, std, min: mn, max: mx });
    }
    return result;
  })();

  useEffect(() => {
    layerStatsRef.current = layerStats;
  }, [layerStats]);

  useEffect(() => {
    if (!canvasRef.current || layerStats.length === 0 || collapsed) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const W = container.offsetWidth || 600;
    const H = 180;
    const PAD = { top: 16, bottom: 32, left: 52, right: 10 };
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

    const d = layerStats.length;

    layoutRef.current = { PAD, plotW, d };

    // Global range
    let globalMin = Infinity, globalMax = -Infinity;
    for (let l = 0; l < d; l++) {
      const s = layerStats[l];
      const lo = Math.min(s.min, s.mean - 2 * s.std);
      const hi = Math.max(s.max, s.mean + 2 * s.std);
      if (lo < globalMin) globalMin = lo;
      if (hi > globalMax) globalMax = hi;
    }
    const absMax = Math.max(Math.abs(globalMin), Math.abs(globalMax), 0.1);
    const rangeMin = -absMax * 1.1;
    const rangeMax = absMax * 1.1;

    const xScale = (l) => PAD.left + (l / Math.max(1, d - 1)) * plotW;
    const yScale = (v) => PAD.top + (1 - (v - rangeMin) / (rangeMax - rangeMin)) * plotH;

    // Draw bands: min/max → ±2σ → ±σ
    const bands = [
      { getY: (s) => [s.min, s.max], alpha: 0.08 },
      { getY: (s) => [s.mean - 2 * s.std, s.mean + 2 * s.std], alpha: 0.14 },
      { getY: (s) => [s.mean - s.std, s.mean + s.std], alpha: 0.26 },
    ];

    for (const band of bands) {
      ctx.beginPath();
      for (let l = 0; l < d; l++) {
        const [lo] = band.getY(layerStats[l]);
        const x = xScale(l);
        l === 0 ? ctx.moveTo(x, yScale(lo)) : ctx.lineTo(x, yScale(lo));
      }
      for (let l = d - 1; l >= 0; l--) {
        const [, hi] = band.getY(layerStats[l]);
        ctx.lineTo(xScale(l), yScale(hi));
      }
      ctx.closePath();
      ctx.fillStyle = `rgba(${FILL_RGB},${band.alpha})`;
      ctx.fill();
    }

    // Mean line
    ctx.beginPath();
    ctx.strokeStyle = LINE_COLOR;
    ctx.lineWidth = 2;
    for (let l = 0; l < d; l++) {
      const x = xScale(l);
      const y = yScale(layerStats[l].mean);
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
      const label = v === 0 ? "0" : v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2);
      ctx.fillText(label, PAD.left - 4, yScale(v) + 3);
    }

    // X-axis labels
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    const labelStep = Math.max(1, Math.floor(d / 10));
    for (let l = 0; l < d; l += labelStep) {
      ctx.fillText(`${l}`, xScale(l), H - 8);
    }
    ctx.fillText("Layer", PAD.left + plotW / 2, H - 20);
  }, [layerStats, collapsed]);

  const handleMouseMove = useCallback((e) => {
    if (!layoutRef.current || layerStatsRef.current.length === 0) return;
    const { PAD, plotW, d } = layoutRef.current;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const frac = (mx - PAD.left) / plotW;
    const layer = Math.round(frac * (d - 1));
    if (layer >= 0 && layer < d) {
      setHover({ layer, pageX: e.pageX, pageY: e.pageY });
    } else {
      setHover(null);
    }
  }, []);

  const handleMouseLeave = useCallback(() => setHover(null), []);

  if (!mlp) return null;

  const hStats = hover ? layerStatsRef.current[hover.layer] : null;

  return (
    <div className="panel">
      <h2
        style={{ cursor: "pointer", userSelect: "none" }}
        onClick={() => setCollapsed((c) => !c)}
      >
        Weight Distributions
        <span style={{ fontSize: 11, color: "#9CA3AF", marginLeft: 8, fontWeight: 400 }}>
          {collapsed ? "▶ show" : "▼ hide"}
        </span>
        {!collapsed && (
          <InfoTip>
            <span className="tip-title">Weight Distributions</span>
            <p className="tip-desc">
              Per-layer weight value distribution shown as mean ±σ band chart across layers.
            </p>
            <div className="tip-sep" />
            <div className="tip-kv"><span className="tip-kv-key">Solid line</span><span className="tip-kv-val">Per-layer mean weight value</span></div>
            <div className="tip-kv"><span className="tip-kv-key">Bands</span><span className="tip-kv-val">±σ, ±2σ, and min–max range</span></div>
            <div className="tip-sep" />
            <p className="tip-desc">
              Stable distributions suggest uniform weight structure; diverging bands indicate layer-dependent patterns.
            </p>
          </InfoTip>
        )}
      </h2>
      {!collapsed && (
        <>
          <div style={{ width: "100%", overflowX: "auto" }}>
            <canvas
              ref={canvasRef}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
              style={{ cursor: "crosshair" }}
            />
          </div>
          <div className="formula-legend" style={{ marginTop: 4 }}>
            <span style={{ color: LINE_COLOR }}>━ mean</span>
            <span style={{ color: `rgba(${FILL_RGB},0.5)` }}>░ ±σ</span>
            <span style={{ color: `rgba(${FILL_RGB},0.25)` }}>░ ±2σ</span>
            <span style={{ color: `rgba(${FILL_RGB},0.12)` }}>░ min–max</span>
          </div>
        </>
      )}
      <CanvasTooltip visible={!!hover && !collapsed} pageX={hover?.pageX} pageY={hover?.pageY}>
        {hStats && (
          <>
            <div className="canvas-tip-header">
              Layer <span className="layer-num">{hover.layer}</span>
            </div>
            <div className="canvas-tip-rows">
              <div className="canvas-tip-row">
                <span className="canvas-tip-label">
                  <span className="canvas-tip-swatch" style={{ background: LINE_COLOR }} />
                  Mean weight
                </span>
                <span className="canvas-tip-value">{hStats.mean.toFixed(4)}</span>
              </div>
              <div className="canvas-tip-row">
                <span className="canvas-tip-label" style={{ paddingLeft: 14 }}>σ</span>
                <span className="canvas-tip-value">
                  ±{hStats.std.toFixed(4)}
                  <span className="canvas-tip-sub">
                    [{hStats.min.toFixed(3)}, {hStats.max.toFixed(3)}]
                  </span>
                </span>
              </div>
            </div>
          </>
        )}
      </CanvasTooltip>
    </div>
  );
}
