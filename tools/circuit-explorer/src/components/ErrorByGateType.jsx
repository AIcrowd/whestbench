/**
 * ErrorByGateType — grouped bar chart showing mean |error| by gate type.
 * Canvas-rendered with CanvasTooltip on hover.
 *
 * Groups gates by their boolean function type (AND, OR, XOR, NOT, BUF, CONST, etc.)
 * and shows how each estimator's error distributes across types.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CanvasTooltip from "./CanvasTooltip";
import InfoTip from "./InfoTip";
import { classifyGate, GATE_TYPE_FONT, GATE_TYPES } from "./gateShapes";

const SERIES = [
  { key: "sampling", label: "Sampling", color: "#F0524D" },
  { key: "meanProp", label: "Mean Prop", color: "#94A3B8" },
  { key: "covProp",  label: "Cov Prop",  color: "#2DD4BF" },
];

export default function ErrorByGateType({
  circuit,
  groundTruth,
  samplingEstimates,
  meanPropEstimates,
  covPropEstimates,
  activeLayer,
}) {
  const canvasRef = useRef(null);
  const layoutRef = useRef(null);
  const [hover, setHover] = useState(null);

  // Compute per-gate-type mean |error| for each estimator
  const chartData = useMemo(() => {
    if (!circuit || !groundTruth) return null;

    const estimators = {};
    if (samplingEstimates) estimators.sampling = samplingEstimates;
    if (meanPropEstimates) estimators.meanProp = meanPropEstimates;
    if (covPropEstimates) estimators.covProp = covPropEstimates;

    if (Object.keys(estimators).length === 0) return null;

    // Determine layer range
    const layers = activeLayer != null ? [activeLayer] : [...Array(circuit.d).keys()];

    // Accumulate errors by gate type
    const groups = {}; // { type: { sampling: { sumErr, count }, ... } }

    for (const l of layers) {
      if (l >= groundTruth.length) continue;
      for (let w = 0; w < circuit.n; w++) {
        const info = classifyGate(circuit.gates[l], w);
        const type = info.type;

        if (!groups[type]) {
          groups[type] = { count: 0 };
          for (const key of Object.keys(estimators)) {
            groups[type][key] = { sumErr: 0, count: 0 };
          }
        }
        groups[type].count++;

        const gt = groundTruth[l][w];
        for (const [key, est] of Object.entries(estimators)) {
          if (l < est.length) {
            const err = Math.abs(est[l][w] - gt);
            groups[type][key].sumErr += err;
            groups[type][key].count++;
          }
        }
      }
    }

    // Convert to sorted array
    const typeOrder = Object.keys(GATE_TYPES);
    const data = typeOrder
      .filter((t) => groups[t])
      .map((type) => {
        const g = groups[type];
        const entry = {
          type,
          symbol: GATE_TYPES[type].symbol,
          color: GATE_TYPES[type].color,
          gateCount: g.count,
        };
        for (const key of Object.keys(estimators)) {
          entry[key] = g[key].count > 0 ? g[key].sumErr / g[key].count : 0;
        }
        return entry;
      });

    return { data, estimatorKeys: Object.keys(estimators) };
  }, [circuit, groundTruth, samplingEstimates, meanPropEstimates, covPropEstimates, activeLayer]);

  // Draw chart
  useEffect(() => {
    if (!canvasRef.current || !chartData || chartData.data.length === 0) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    const W = container.offsetWidth || 600;
    const H = 220;
    const PAD = { top: 14, bottom: 42, left: 52, right: 10 };
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

    const { data, estimatorKeys } = chartData;
    const nGroups = data.length;
    const nBars = estimatorKeys.length;

    // Find max value for y-axis
    let maxVal = 0;
    for (const entry of data) {
      for (const key of estimatorKeys) {
        if (entry[key] > maxVal) maxVal = entry[key];
      }
    }
    maxVal = maxVal * 1.15 || 0.01;

    const groupW = plotW / nGroups;
    const barW = Math.min(groupW * 0.7 / nBars, 30);
    const barGap = Math.max(2, barW * 0.15);
    const totalBarGroupW = nBars * barW + (nBars - 1) * barGap;

    const yScale = (v) => PAD.top + (1 - v / maxVal) * plotH;

    // Store layout for hover
    layoutRef.current = {
      PAD, plotW, nGroups, nBars, groupW, barW, barGap, totalBarGroupW,
      estimatorKeys, data, maxVal,
    };

    // Grid lines
    ctx.strokeStyle = "rgba(156,163,175,0.2)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const v = (maxVal / yTicks) * i;
      const y = yScale(v);
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + plotW, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw bars
    for (let g = 0; g < nGroups; g++) {
      const entry = data[g];
      const groupCenterX = PAD.left + g * groupW + groupW / 2;
      const groupStartX = groupCenterX - totalBarGroupW / 2;

      for (let b = 0; b < nBars; b++) {
        const key = estimatorKeys[b];
        const val = entry[key];
        const barX = groupStartX + b * (barW + barGap);
        const barH = (val / maxVal) * plotH;
        const barY = PAD.top + plotH - barH;

        const series = SERIES.find((s) => s.key === key);
        ctx.fillStyle = series ? series.color : "#94A3B8";
        ctx.globalAlpha = 0.85;
        ctx.fillRect(barX, barY, barW, barH);
        ctx.globalAlpha = 1;

        // Bar outline
        ctx.strokeStyle = series ? series.color : "#94A3B8";
        ctx.lineWidth = 1;
        ctx.strokeRect(barX, barY, barW, barH);
      }
    }

    // X-axis labels (gate type symbol + name)
    ctx.fillStyle = "#64748B";
    ctx.font = `bold 12px ${GATE_TYPE_FONT}`;
    ctx.textAlign = "center";
    for (let g = 0; g < nGroups; g++) {
      const cx = PAD.left + g * groupW + groupW / 2;
      const entry = data[g];
      // Symbol
      ctx.fillStyle = entry.color;
      ctx.fillText(entry.symbol, cx, H - 20);
      // Type name
      ctx.fillStyle = entry.color;
      ctx.font = `9px ${GATE_TYPE_FONT}`;
      ctx.fillText(entry.type, cx, H - 6);
      ctx.font = `bold 12px ${GATE_TYPE_FONT}`;
    }

    // Y-axis labels
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "right";
    for (let i = 0; i <= yTicks; i++) {
      const v = (maxVal / yTicks) * i;
      const label = v < 0.001 && v !== 0 ? v.toExponential(1) : v.toFixed(3);
      ctx.fillText(label, PAD.left - 4, yScale(v) + 3);
    }
  }, [chartData]);

  // Hover handler
  const handleMouseMove = useCallback((e) => {
    if (!layoutRef.current || !chartData) return;
    const { PAD, nGroups, groupW, data } = layoutRef.current;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const groupIdx = Math.floor((mx - PAD.left) / groupW);
    if (groupIdx >= 0 && groupIdx < nGroups) {
      setHover({ groupIdx, pageX: e.pageX, pageY: e.pageY, entry: data[groupIdx] });
    } else {
      setHover(null);
    }
  }, [chartData]);

  const handleMouseLeave = useCallback(() => setHover(null), []);

  if (!chartData || chartData.data.length === 0) return null;

  const fmtErr = (v) => v < 0.0001 && v !== 0 ? v.toExponential(4) : v.toFixed(4);

  return (
    <div className="panel">
      <h2>
        Error by Gate Type
        {activeLayer != null && (
          <span className="mode-badge">Layer {activeLayer}</span>
        )}
        <InfoTip>
          <span className="tip-title">Error by Gate Type</span>
          <p className="tip-desc">
            Mean <span className="tip-mono">|error|</span> grouped by boolean gate type{activeLayer != null ? ` at layer ${activeLayer}` : " across all layers"}.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Insight</span><span className="tip-kv-val">Shows which gate types are harder for each estimator</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Interact</span><span className="tip-kv-val">Click a layer in the MSE chart to filter</span></div>
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
        {chartData.estimatorKeys.map((key) => {
          const s = SERIES.find((s) => s.key === key);
          return s ? (
            <span key={key} style={{ color: s.color }}>■ {s.label}</span>
          ) : null;
        })}
      </div>
      <CanvasTooltip visible={!!hover} pageX={hover?.pageX} pageY={hover?.pageY}>
        {hover?.entry && (
          <>
            <div className="canvas-tip-header">
              <span style={{ color: hover.entry.color, fontWeight: 700, marginRight: 4, fontFamily: GATE_TYPE_FONT }}>
                {hover.entry.symbol}
              </span>
              <span style={{ fontFamily: GATE_TYPE_FONT, color: hover.entry.color }}>{hover.entry.type}</span>
              <span style={{ fontWeight: 400, color: "#9CA3AF", marginLeft: 6, fontSize: 10 }}>
                ({hover.entry.gateCount} gates)
              </span>
            </div>
            <div className="canvas-tip-rows">
              {chartData.estimatorKeys.map((key) => {
                const s = SERIES.find((s) => s.key === key);
                return (
                  <div key={key} className="canvas-tip-row">
                    <span className="canvas-tip-label">
                      <span className="canvas-tip-swatch" style={{ background: s?.color || "#94A3B8" }} />
                      {s?.label || key}
                    </span>
                    <span className="canvas-tip-value">{fmtErr(hover.entry[key])}</span>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </CanvasTooltip>
    </div>
  );
}
