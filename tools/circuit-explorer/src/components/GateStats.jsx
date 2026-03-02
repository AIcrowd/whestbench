/**
 * GateStats — Canvas-rendered stacked bar chart.
 * Shows per-layer dominant coefficient composition as a stacked bar.
 * One bar per layer, each segment colored by coefficient type.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CanvasTooltip from "./CanvasTooltip";
import InfoTip from "./InfoTip";

/* ── Strict palette from circuit gate colors ── */
const COLORS = {
  c: "#8B95A2",   // bias — medium gray
  a: "#5B7BA8",   // first input — steel blue
  b: "#1E293B",   // second input — near-black slate
  p: "#F0524D",   // product — coral
};

const COEFF_META = {
  c: { label: "constant bias",     fill: COLORS.c },
  a: { label: "first input (x)",   fill: COLORS.a },
  b: { label: "second input (y)",  fill: COLORS.b },
  p: { label: "interaction (x·y)", fill: COLORS.p },
};
const KEYS = ["c", "a", "b", "p"];

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
      let cCount = 0, aCount = 0, bCount = 0, pCount = 0;
      for (let i = 0; i < n; i++) {
        const cv = Math.abs(layer.const[i]);
        const av = Math.abs(layer.firstCoeff[i]);
        const bv = Math.abs(layer.secondCoeff[i]);
        const pv = Math.abs(layer.productCoeff[i]);
        const max = Math.max(cv, av, bv, pv);
        if (max === pv) pCount++;
        else if (max === cv) cCount++;
        else if (max === av) aCount++;
        else bCount++;
      }
      result.push({ c: cCount, a: aCount, b: bCount, p: pCount, total: n });
    }
    return result;
  }, [circuit]);

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

      for (const key of KEYS) {
        const count = data[key];
        const segH = (count / maxVal) * plotH;
        yStack -= segH;
        ctx.fillStyle = COLORS[key];
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
  }, [layerData, activeLayer]);

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
            Stacked bar chart showing which coefficient type dominates each gate per layer.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Each bar</span><span className="tip-kv-val">Segments gates by largest absolute coefficient</span></div>
          <div className="tip-kv"><span className="tip-kv-key"><span className="tip-mono">c</span></span><span className="tip-kv-val">Constant bias</span></div>
          <div className="tip-kv"><span className="tip-kv-key"><span className="tip-mono">a</span>, <span className="tip-mono">b</span></span><span className="tip-kv-val">First / second input weights</span></div>
          <div className="tip-kv"><span className="tip-kv-key"><span className="tip-mono">p</span></span><span className="tip-kv-val">Product interaction (x·y)</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            <span className="tip-highlight">Product-heavy</span> layers are harder to estimate — x·y creates non-linear wire dependencies.
          </p>
        </InfoTip>
      </h2>
      <div className="gate-formula-explain">
        <code className="formula-box">out = c + a·x + b·y + p·x·y</code>
      </div>
      <div style={{ width: "100%", overflowX: "auto" }}>
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ cursor: "crosshair" }}
        />
      </div>
      <div className="formula-legend" style={{ marginTop: 4 }}>
        {KEYS.map(k => (
          <span key={k} style={{ color: COEFF_META[k].fill }}>
            ■ <strong>{k}</strong> {COEFF_META[k].label}
          </span>
        ))}
      </div>
      <p className="panel-desc">
        Each gate's output depends on inputs <em>x</em>, <em>y</em> ∈ {"{"}−1, +1{"}"}.
        <strong> Product-heavy</strong> circuits are harder to estimate because
        the x·y interaction creates non-linear dependencies between wires.
      </p>
      <CanvasTooltip visible={!!hover} pageX={hover?.pageX} pageY={hover?.pageY}>
        {hData && (
          <>
            <div className="canvas-tip-header">
              Layer <span className="layer-num">{hover.layer}</span>
            </div>
            <div className="canvas-tip-rows">
              {KEYS.map(k => (
                <div className="canvas-tip-row" key={k}>
                  <span className="canvas-tip-label">
                    <span className="canvas-tip-swatch" style={{ background: COLORS[k] }} />
                    {k} — {COEFF_META[k].label}
                  </span>
                  <span className="canvas-tip-value">
                    {hData[k]} gates
                    <span className="canvas-tip-sub">
                      ({(hData[k] / hData.total * 100).toFixed(1)}%)
                    </span>
                  </span>
                </div>
              ))}
            </div>
          </>
        )}
      </CanvasTooltip>
    </div>
  );
}
