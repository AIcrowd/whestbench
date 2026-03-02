/**
 * GateStats — Canvas-rendered stacked bar chart.
 * Shows per-layer dominant coefficient composition as a stacked bar.
 * One bar per layer, each segment colored by coefficient type.
 */
import { useEffect, useMemo, useRef } from "react";

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

  if (!circuit) return null;

  return (
    <div className="panel">
      <h2>Gate Structure Analysis</h2>
      <div className="gate-formula-explain">
        <code className="formula-box">out = c + a·x + b·y + p·x·y</code>
      </div>
      <div style={{ width: "100%", overflowX: "auto" }}>
        <canvas ref={canvasRef} />
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
    </div>
  );
}
