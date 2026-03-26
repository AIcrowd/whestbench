/**
 * EstimatorComparison — per-layer MSE comparison.
 * Canvas-rendered bar/line chart with CanvasTooltip on hover.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CanvasTooltip from "./CanvasTooltip";
import InfoTip from "./InfoTip";

const SERIES_COLORS = {
  sampling:  "#F0524D",  // Coral (+1)
  covProp:   "#B29F9E",  // Blended midpoint/neutral
  meanProp:  "#334155",  // Dark Slate (-1)
};

export default function EstimatorComparison({
  groundTruth,
  samplingEstimates,
  meanPropEstimates,
  covPropEstimates,
  depth,
  activeLayer,
}) {
  const canvasRef = useRef(null);
  const layoutRef = useRef(null);
  const [hover, setHover] = useState(null);

  const mseData = useMemo(() => {
    if (!groundTruth) return null;
    const hasSampling = !!samplingEstimates;
    const hasMeanProp = !!meanPropEstimates;
    const hasCovProp = !!covPropEstimates;
    const layers = Math.min(
      depth,
      groundTruth.length,
      hasSampling ? samplingEstimates.length : Infinity,
      hasMeanProp ? meanPropEstimates.length : Infinity,
      hasCovProp ? covPropEstimates.length : Infinity
    );

    const result = [];
    for (let l = 0; l < layers; l++) {
      const n = groundTruth[l].length;
      const entry = { layer: l };
      if (hasSampling) {
        let mse = 0;
        for (let i = 0; i < n; i++) {
          mse += (samplingEstimates[l][i] - groundTruth[l][i]) ** 2;
        }
        entry.sampling = mse / n;
      }
      if (hasMeanProp) {
        let mse = 0;
        for (let i = 0; i < n; i++) {
          mse += (meanPropEstimates[l][i] - groundTruth[l][i]) ** 2;
        }
        entry.meanProp = mse / n;
      }
      if (hasCovProp) {
        let mse = 0;
        for (let i = 0; i < n; i++) {
          mse += (covPropEstimates[l][i] - groundTruth[l][i]) ** 2;
        }
        entry.covProp = mse / n;
      }
      result.push(entry);
    }
    return result;
  }, [groundTruth, samplingEstimates, meanPropEstimates, covPropEstimates, depth]);

  useEffect(() => {
    if (!canvasRef.current || !mseData || mseData.length === 0) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const W = container.offsetWidth || 600;
    const H = 240;
    const PAD = { top: 14, bottom: 28, left: 52, right: 10 };
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

    const d = mseData.length;
    const hasSampling = mseData[0].sampling !== undefined;
    const hasMeanProp = mseData[0].meanProp !== undefined;
    const hasCovProp = mseData[0].covProp !== undefined;

    // Find max MSE for y-axis
    let maxMSE = 0;
    for (const entry of mseData) {
      if (hasSampling && entry.sampling > maxMSE) maxMSE = entry.sampling;
      if (hasMeanProp && entry.meanProp > maxMSE) maxMSE = entry.meanProp;
      if (hasCovProp && entry.covProp > maxMSE) maxMSE = entry.covProp;
    }
    maxMSE = maxMSE * 1.15 || 0.01; // 15% headroom

    const xScale = (l) => PAD.left + (l / Math.max(1, d - 1)) * plotW;
    const yScale = (v) => PAD.top + (1 - v / maxMSE) * plotH;

    // Store layout for hover
    layoutRef.current = { PAD, plotW, d, hasSampling, hasMeanProp, hasCovProp };

    // Grid lines
    ctx.strokeStyle = "rgba(156,163,175,0.2)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const v = (maxMSE / yTicks) * i;
      const y = yScale(v);
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + plotW, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw lines and dots
    const drawSeries = (key, color) => {
      // Line
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (let l = 0; l < d; l++) {
        const x = xScale(l);
        const y = yScale(mseData[l][key]);
        l === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Dots
      ctx.fillStyle = color;
      for (let l = 0; l < d; l++) {
        const x = xScale(l);
        const y = yScale(mseData[l][key]);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    if (hasSampling) drawSeries("sampling", SERIES_COLORS.sampling);
    if (hasMeanProp) drawSeries("meanProp", SERIES_COLORS.meanProp);
    if (hasCovProp) drawSeries("covProp", SERIES_COLORS.covProp);

    // Active layer highlight
    if (activeLayer != null && activeLayer >= 0 && activeLayer < d) {
      const x = xScale(activeLayer);
      ctx.beginPath();
      ctx.strokeStyle = "rgba(240,82,77,0.5)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.moveTo(x, PAD.top);
      ctx.lineTo(x, PAD.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // X-axis labels
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    const labelStep = Math.max(1, Math.floor(d / 10));
    for (let l = 0; l < d; l += labelStep) {
      ctx.fillText(`${l}`, xScale(l), H - 4);
    }
    ctx.fillText("Layer", PAD.left + plotW / 2, H - 14);

    // Y-axis labels
    ctx.textAlign = "right";
    for (let i = 0; i <= yTicks; i++) {
      const v = (maxMSE / yTicks) * i;
      const label = v < 0.0001 && v !== 0 ? v.toExponential(1) : v.toFixed(4);
      ctx.fillText(label, PAD.left - 4, yScale(v) + 3);
    }
  }, [mseData, activeLayer]);

  const handleMouseMove = useCallback((e) => {
    if (!layoutRef.current || !mseData || mseData.length === 0) return;
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
  }, [mseData]);

  const handleMouseLeave = useCallback(() => setHover(null), []);

  if (!groundTruth) return null;

  const fmtMSE = (v) => v < 0.0001 && v !== 0 ? v.toExponential(4) : v.toFixed(4);
  const hData = hover && mseData ? mseData[hover.layer] : null;

  return (
    <div className="panel">
      <h2>
        Estimation Error (MSE per Layer)
        <InfoTip>
          <span className="tip-title">Estimation Error (MSE)</span>
          <p className="tip-desc">
            Per-layer <span className="tip-mono">Mean Squared Error</span> between each estimator's E[neuron] predictions and the ground truth.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Lower</span><span className="tip-kv-val">Better estimation accuracy</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Trend</span><span className="tip-kv-val">Error typically grows with depth</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            Approximation assumptions (e.g. neuron independence in mean propagation) <span className="tip-highlight">compound across layers</span>.
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
        {!!samplingEstimates && (
          <span style={{ color: SERIES_COLORS.sampling }}>● Sampling Error</span>
        )}
        {!!meanPropEstimates && (
          <span style={{ color: SERIES_COLORS.meanProp }}>● Mean Prop Error</span>
        )}
        {!!covPropEstimates && (
          <span style={{ color: SERIES_COLORS.covProp }}>● Cov Prop Error</span>
        )}
      </div>
      <CanvasTooltip visible={!!hover} pageX={hover?.pageX} pageY={hover?.pageY}>
        {hData && (
          <>
            <div className="canvas-tip-header">
              Layer <span className="layer-num">{hover.layer}</span>
            </div>
            <div className="canvas-tip-rows">
              {hData.sampling !== undefined && (
                <div className="canvas-tip-row">
                  <span className="canvas-tip-label">
                    <span className="canvas-tip-swatch" style={{ background: SERIES_COLORS.sampling }} />
                    Sampling MSE
                  </span>
                  <span className="canvas-tip-value">{fmtMSE(hData.sampling)}</span>
                </div>
              )}
              {hData.meanProp !== undefined && (
                <div className="canvas-tip-row">
                  <span className="canvas-tip-label">
                    <span className="canvas-tip-swatch" style={{ background: SERIES_COLORS.meanProp }} />
                    Mean Prop MSE
                  </span>
                  <span className="canvas-tip-value">{fmtMSE(hData.meanProp)}</span>
                </div>
              )}
              {hData.covProp !== undefined && (
                <div className="canvas-tip-row">
                  <span className="canvas-tip-label">
                    <span className="canvas-tip-swatch" style={{ background: SERIES_COLORS.covProp }} />
                    Cov Prop MSE
                  </span>
                  <span className="canvas-tip-value">{fmtMSE(hData.covProp)}</span>
                </div>
              )}
            </div>
          </>
        )}
      </CanvasTooltip>
    </div>
  );
}
