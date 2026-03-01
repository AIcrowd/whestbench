import * as tfvis from "@tensorflow/tfjs-vis";
import { useEffect, useMemo, useRef } from "react";
import { perfEnd, perfStart } from "../perf";

/* ── Strict palette from circuit gate colors (gateShapes.js) ── */
const COLORS = {
  const:   "#D1D5DB",
  first:   "#94A3B8",
  second:  "#334155",
  product: "#F0524D",
};

/* Unified coefficient labels */
const COEFF_META = {
  c: { label: "constant bias",    fill: COLORS.const },
  a: { label: "first input (x)",  fill: COLORS.first },
  b: { label: "second input (y)", fill: COLORS.second },
  p: { label: "interaction (x·y)", fill: COLORS.product },
};

const MAX_LAYER_BARS = 64;

export default function GateStats({ circuit, activeLayer }) {
  const dominantRef = useRef(null);
  const magnitudeRef = useRef(null);

  const { layerData, coeffData, summaryData, showPerLayerChart } = useMemo(() => {
    if (!circuit) return { layerData: [], coeffData: [], summaryData: [], showPerLayerChart: true };
    perfStart('gatestats-compute');
    const { n, d, gates } = circuit;

    const _layerData = [];
    let totalC = 0, totalA = 0, totalB = 0, totalP = 0;

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

      totalC += cCount; totalA += aCount; totalB += bCount; totalP += pCount;
      _layerData.push({
        layer: `L${l}`,
        c: cCount, a: aCount, b: bCount, p: pCount,
      });
    }

    // Overall coefficient magnitude stats
    const totals = { const: 0, first: 0, second: 0, product: 0 };
    let count = 0;
    for (let l = 0; l < d; l++) {
      const layer = gates[l];
      for (let i = 0; i < n; i++) {
        totals.const += Math.abs(layer.const[i]);
        totals.first += Math.abs(layer.firstCoeff[i]);
        totals.second += Math.abs(layer.secondCoeff[i]);
        totals.product += Math.abs(layer.productCoeff[i]);
        count++;
      }
    }

    const _coeffData = [
      { index: "c — bias",        value: totals.const / count },
      { index: "a — first (x)",   value: totals.first / count },
      { index: "b — second (y)",  value: totals.second / count },
      { index: "p — product (xy)", value: totals.product / count },
    ];

    const totalGates = totalC + totalA + totalB + totalP;
    const _summaryData = [
      { index: "c — bias",        value: (totalC / totalGates * 100) },
      { index: "a — first (x)",   value: (totalA / totalGates * 100) },
      { index: "b — second (y)",  value: (totalB / totalGates * 100) },
      { index: "p — product (xy)", value: (totalP / totalGates * 100) },
    ];

    perfEnd('gatestats-compute');
    return {
      layerData: _layerData,
      coeffData: _coeffData,
      summaryData: _summaryData,
      showPerLayerChart: d <= MAX_LAYER_BARS,
    };
  }, [circuit]);

  // Render dominant coefficient chart
  useEffect(() => {
    if (!dominantRef.current || !circuit) return;
    dominantRef.current.innerHTML = '';

    if (showPerLayerChart) {
      // For small circuits — per-layer stacked data as separate series
      // tfvis barchart only does single series, so we use a grouped approach
      // rendering one bar chart per coefficient type
      const data = layerData.map(d => ({
        index: d.layer,
        value: d.c + d.a + d.b + d.p,
      }));
      tfvis.render.barchart(dominantRef.current, data, {
        width: dominantRef.current.offsetWidth || 400,
        height: 180,
        xLabel: 'Layer',
        yLabel: 'Gates',
        color: COLORS.product,
      });
    } else {
      // For large circuits — summary distribution
      tfvis.render.barchart(dominantRef.current, summaryData, {
        width: dominantRef.current.offsetWidth || 400,
        height: 180,
        xLabel: 'Coefficient',
        yLabel: '% of gates',
        color: [COLORS.const, COLORS.first, COLORS.second, COLORS.product],
      });
    }
  }, [circuit, layerData, summaryData, showPerLayerChart]);

  // Render coefficient magnitude chart
  useEffect(() => {
    if (!magnitudeRef.current || !circuit) return;
    magnitudeRef.current.innerHTML = '';

    tfvis.render.barchart(magnitudeRef.current, coeffData, {
      width: magnitudeRef.current.offsetWidth || 400,
      height: 210,
      xLabel: 'Coefficient',
      yLabel: 'Avg |value|',
      color: [COLORS.const, COLORS.first, COLORS.second, COLORS.product],
    });
  }, [circuit, coeffData]);

  if (!circuit) return null;

  return (
    <div className="panel">
      <h2>Gate Structure Analysis</h2>

      {/* Explanation of the gate formula */}
      <div className="gate-formula-explain">
        <code className="formula-box">out = c + a·x + b·y + p·x·y</code>
        <div className="formula-legend">
          <span style={{ color: COLORS.const }}>● <strong>c</strong> constant bias</span>
          <span style={{ color: COLORS.first }}>● <strong>a</strong> first input (x)</span>
          <span style={{ color: COLORS.second }}>● <strong>b</strong> second input (y)</span>
          <span style={{ color: COLORS.product }}>● <strong>p</strong> interaction (x·y)</span>
        </div>
      </div>

      <div className="gate-stats-grid">
        <div>
          <h3 className="subheading">
            {showPerLayerChart ? "Dominant Coefficient per Layer" : "Dominant Coefficient Distribution"}
          </h3>
          <div ref={dominantRef} style={{ width: '100%', minHeight: 180 }} />
        </div>

        <div>
          <h3 className="subheading">Avg |Coefficient| Magnitude</h3>
          <div ref={magnitudeRef} style={{ width: '100%', minHeight: 210 }} />
        </div>
      </div>

      <p className="panel-desc">
        Each gate's output depends on inputs <em>x</em>, <em>y</em> ∈ {"{"}−1, +1{"}"}.
        <strong> Product-heavy</strong> circuits are harder to estimate because
        the x·y interaction creates non-linear dependencies between wires.
      </p>
    </div>
  );
}
