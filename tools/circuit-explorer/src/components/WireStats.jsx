/**
 * WireStats — Per-layer E[wire] distribution as a band chart.
 * Shows mean ±σ, ±2σ as nested colored bands (same style as ActivationRibbon).
 * Canvas-rendered for performance.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import CanvasTooltip from "./CanvasTooltip";
import InfoTip from "./InfoTip";

export default function WireStats({ means, width: n, depth: d, source, activeLayer }) {
  const canvasRef = useRef(null);
  const layoutRef = useRef(null);
  const aggRef = useRef([]);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    if (!means || means.length === 0 || !canvasRef.current) return;
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

    // Store layout for hover
    layoutRef.current = { PAD, plotW, layers };

    // Compute per-layer aggregate stats from wire means
    const agg = [];
    for (let l = 0; l < layers; l++) {
      let sum = 0, sumSq = 0, mn = Infinity, mx = -Infinity;
      for (let w = 0; w < n && w < means[l].length; w++) {
        const val = means[l][w] || 0;
        sum += val;
        sumSq += val * val;
        if (val < mn) mn = val;
        if (val > mx) mx = val;
      }
      const avg = sum / n;
      const variance = sumSq / n - avg * avg;
      const std = Math.sqrt(Math.max(0, variance));
      agg.push({ mean: avg, std, min: mn, max: mx });
    }
    aggRef.current = agg;

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

    // Active layer highlight
    if (activeLayer != null && activeLayer >= 0 && activeLayer < layers) {
      const x = xScale(activeLayer);
      ctx.beginPath();
      ctx.strokeStyle = "rgba(240,82,77,0.5)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.moveTo(x, PAD.top);
      ctx.lineTo(x, H - PAD.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

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
  }, [means, n, d, activeLayer]);

  const handleMouseMove = useCallback((e) => {
    if (!layoutRef.current || aggRef.current.length === 0) return;
    const { PAD, plotW, layers } = layoutRef.current;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const frac = (mx - PAD.left) / plotW;
    const layer = Math.round(frac * (layers - 1));
    if (layer >= 0 && layer < layers) {
      setHover({ layer, pageX: e.pageX, pageY: e.pageY });
    } else {
      setHover(null);
    }
  }, []);

  const handleMouseLeave = useCallback(() => setHover(null), []);

  if (!means || means.length === 0) return null;

  const hAgg = hover ? aggRef.current[hover.layer] : null;

  return (
    <div className="panel">
      <h2>
        E[wire] Distribution <small>(across wires)</small>
        <InfoTip>
          <span className="tip-title">E[wire] Distribution</span>
          <p className="tip-desc">
            Each wire has an expected output <span className="tip-mono">E[wire]</span> — its average over all possible inputs. This plot shows how those values spread across wires within each layer.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Solid line</span><span className="tip-kv-val">Layer-wide average E[wire]</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Bands</span><span className="tip-kv-val">±1σ, ±2σ, and min–max of wire values</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            <span className="tip-highlight">Wide bands</span> → wires have diverse biases.{" "}
            <span className="tip-highlight">Narrow bands</span> → all wires biased similarly.
          </p>
        </InfoTip>
        {source && <span className="source-badge">{source}</span>}
      </h2>
      <div style={{ width: '100%', overflowX: 'auto' }}>
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ cursor: "crosshair" }}
        />
      </div>
      <div className="formula-legend" style={{ marginTop: 4 }}>
        <span style={{ color: "#F0524D" }}>━ mean</span>
        <span style={{ color: "rgba(240,82,77,0.5)" }}>░ ±σ</span>
        <span style={{ color: "rgba(240,82,77,0.3)" }}>░ ±2σ</span>
        <span style={{ color: "rgba(240,82,77,0.15)" }}>░ min–max</span>
      </div>
      <CanvasTooltip visible={!!hover} pageX={hover?.pageX} pageY={hover?.pageY}>
        {hAgg && (
          <>
            <div className="canvas-tip-header">
              Layer <span className="layer-num">{hover.layer}</span>
            </div>
            <div className="canvas-tip-rows">
              <div className="canvas-tip-row">
                <span className="canvas-tip-label">
                  <span className="canvas-tip-swatch" style={{ background: "#F0524D" }} />
                  Mean E[wire]
                </span>
                <span className="canvas-tip-value">{hAgg.mean.toFixed(4)}</span>
              </div>
              <div className="canvas-tip-row">
                <span className="canvas-tip-label">
                  <span className="canvas-tip-swatch" style={{ background: "rgba(240,82,77,0.5)" }} />
                  Std dev (σ)
                </span>
                <span className="canvas-tip-value">±{hAgg.std.toFixed(4)}</span>
              </div>
              <div className="canvas-tip-divider" />
              <div className="canvas-tip-row">
                <span className="canvas-tip-label">
                  <span className="canvas-tip-swatch" style={{ background: "rgba(240,82,77,0.22)" }} />
                  ±1σ range
                </span>
                <span className="canvas-tip-value">
                  [{(hAgg.mean - hAgg.std).toFixed(4)}, {(hAgg.mean + hAgg.std).toFixed(4)}]
                </span>
              </div>
              <div className="canvas-tip-row">
                <span className="canvas-tip-label">
                  <span className="canvas-tip-swatch" style={{ background: "rgba(240,82,77,0.08)" }} />
                  Min / Max
                </span>
                <span className="canvas-tip-value">
                  [{hAgg.min.toFixed(4)}, {hAgg.max.toFixed(4)}]
                </span>
              </div>
            </div>
          </>
        )}
      </CanvasTooltip>
    </div>
  );
}
