/**
 * GateStats — Canvas-rendered stacked bar chart.
 * Shows per-layer gate type distribution as a stacked bar.
 * One bar per layer, each segment colored by boolean gate type.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CanvasTooltip from "./CanvasTooltip";
import InfoTip from "./InfoTip";
import { classifyGate, GATE_TYPE_FONT, GATE_TYPES } from "./gateShapes";

export default function GateStats({ circuit, activeLayer }) {
  const canvasRef = useRef(null);
  const layoutRef = useRef(null);
  const [hover, setHover] = useState(null);

  const layerData = useMemo(() => {
    if (!circuit) return [];
    const { n, d, gates } = circuit;
    const result = [];
    for (let l = 0; l < d; l++) {
      const layer = gates[l];
      const counts = {};
      Object.keys(GATE_TYPES).forEach(k => counts[k] = 0);
      for (let w = 0; w < n; w++) {
        const info = classifyGate(layer, w);
        counts[info.type] = (counts[info.type] || 0) + 1;
      }
      result.push({ ...counts, total: n });
    }
    return result;
  }, [circuit]);

  const availableTypes = useMemo(() => {
    if (!layerData.length) return [];
    return Object.keys(GATE_TYPES).filter(k => layerData.some(d => d[k] > 0));
  }, [layerData]);

  useEffect(() => {
    if (!canvasRef.current || layerData.length === 0) return;
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

    const d = layerData.length;
    const barGap = d > 50 ? 0 : 1;
    const barW = Math.max(1, (plotW - (d - 1) * barGap) / d);
    const maxVal = layerData[0].total; // all layers have n gates

    // Store layout for hover detection
    layoutRef.current = { PAD, barW, barGap, d, plotW };

    for (let l = 0; l < d; l++) {
      const x = PAD.left + l * (barW + barGap);
      let yStack = PAD.top + plotH;
      const data = layerData[l];
      const isActive = l === activeLayer;

      for (const key of availableTypes) {
        const count = data[key];
        const segH = (count / maxVal) * plotH;
        yStack -= segH;
        ctx.fillStyle = GATE_TYPES[key].color;
        ctx.globalAlpha = isActive ? 1.0 : 0.85;
        ctx.fillRect(x, yStack, barW, segH);
      }

      // Active layer highlight
      if (isActive) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "#F0524D";
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 0.5, PAD.top, barW + 1, plotH);
      }
      ctx.globalAlpha = 1;
    }

    // X-axis labels
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    const labelStep = Math.max(1, Math.floor(d / 10));
    for (let l = 0; l < d; l += labelStep) {
      const x = PAD.left + l * (barW + barGap) + barW / 2;
      ctx.fillText(`${l}`, x, H - 4);
    }
    ctx.fillText("Layer", PAD.left + plotW / 2, H - 14);

    // Y-axis labels (percentage)
    ctx.textAlign = "right";
    ctx.fillText("100%", PAD.left - 4, PAD.top + 4);
    ctx.fillText("50%", PAD.left - 4, PAD.top + plotH / 2 + 3);
    ctx.fillText("0%", PAD.left - 4, PAD.top + plotH + 3);
  }, [layerData, activeLayer, availableTypes]);

  const handleMouseMove = useCallback((e) => {
    if (!layoutRef.current || layerData.length === 0) return;
    const { PAD, barW, barGap, d } = layoutRef.current;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const layer = Math.floor((mx - PAD.left) / (barW + barGap));
    if (layer >= 0 && layer < d) {
      setHover({ layer, pageX: e.pageX, pageY: e.pageY });
    } else {
      setHover(null);
    }
  }, [layerData]);

  const handleMouseLeave = useCallback(() => setHover(null), []);

  if (!circuit) return null;

  const hData = hover ? layerData[hover.layer] : null;

  return (
    <div className="panel">
      <h2>
        Gate Structure Analysis
        <InfoTip>
          <span className="tip-title">Gate Structure</span>
          <p className="tip-desc">
            Stacked bar chart showing the distribution of Boolean gate types per layer.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Each bar</span><span className="tip-kv-val">Shows the proportion of different gate types (e.g., AND, XOR) in that layer</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            Gate types with non-linear cross-terms like <span className="tip-highlight">AND / OR</span> are harder to estimate than linear gates like <span className="tip-highlight">XOR</span>.
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
      <div className="formula-legend" style={{ marginTop: 4, display: "flex", flexWrap: "wrap", gap: "8px" }}>
        {availableTypes.map(k => (
          <span key={k} style={{ color: GATE_TYPES[k].color }}>
            ■ <strong style={{ fontFamily: GATE_TYPE_FONT }}>{GATE_TYPES[k].symbol}</strong> {k}
          </span>
        ))}
      </div>
      <p className="panel-desc">
        Visualizes the composition of boolean gates across the circuit's depth.
        Different gate types pose distinct challenges for estimators (e.g., non-linear vs. linear gates).
      </p>
      <CanvasTooltip visible={!!hover} pageX={hover?.pageX} pageY={hover?.pageY}>
        {hData && (
          <>
            <div className="canvas-tip-header">
              Layer <span className="layer-num">{hover.layer}</span>
            </div>
            <div className="canvas-tip-rows">
              {availableTypes.slice().sort((a, b) => (hData[b] || 0) - (hData[a] || 0)).map(k => {
                if (hData[k] === 0) return null;
                return (
                  <div className="canvas-tip-row" key={k}>
                    <span className="canvas-tip-label">
                      <span className="canvas-tip-swatch" style={{ background: GATE_TYPES[k].color }} />
                      <span style={{ fontFamily: GATE_TYPE_FONT, marginRight: 4, fontWeight: 'bold', color: GATE_TYPES[k].color }}>
                        {GATE_TYPES[k].symbol}
                      </span>
                      {k}
                    </span>
                    <span className="canvas-tip-value">
                      {hData[k]} gates
                      <span className="canvas-tip-sub">
                        ({(hData[k] / hData.total * 100).toFixed(1)}%)
                      </span>
                    </span>
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
