import { useMemo } from "react";
import {
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from "recharts";
import { perfEnd, perfStart } from "../perf";

/* ── Strict palette from circuit gate colors (gateShapes.js) ── */
const COLORS = {
  const:   "#D1D5DB", // constant gate border
  first:   "#94A3B8", // linear gate border (slate)
  second:  "#334155", // dark slate (mean = -1 anchor)
  product: "#F0524D", // coral (product / and gate)
  mixed:   "#F7A09D", // coral-light tint
};

/* Unified coefficient labels */
const COEFF_META = {
  c: { label: "constant bias",    fill: COLORS.const },
  a: { label: "first input (x)",  fill: COLORS.first },
  b: { label: "second input (y)", fill: COLORS.second },
  p: { label: "interaction (x·y)", fill: COLORS.product },
};

/* Max layers for per-layer SVG bar chart (avoid 1000+ rect nodes) */
const MAX_LAYER_BARS = 64;

/* Custom x-axis tick — color-coded coefficient key + description */
function ColoredTick({ x, y, payload }) {
  const meta = COEFF_META[payload.value];
  if (!meta) return null;
  return (
    <g transform={`translate(${x},${y})`}>
      <text x={0} y={0} dy={12} textAnchor="middle" fontSize={11}
        fontWeight={700} fontFamily="'IBM Plex Mono', monospace" fill={meta.fill}>
        {payload.value}
      </text>
      <text x={0} y={0} dy={24} textAnchor="middle" fontSize={9}
        fill="#9CA3AF" fontFamily="'DM Sans', sans-serif">
        {meta.label}
      </text>
    </g>
  );
}

export default function GateStats({ circuit, activeLayer }) {
  // Memoize all data computation — avoid recomputing 262k entries per render
  const { layerData, coeffData, summaryData, showPerLayerChart } = useMemo(() => {
    if (!circuit) return { layerData: [], coeffData: [], summaryData: [], showPerLayerChart: true };
    perfStart('gatestats-compute');
    const { n, d, gates } = circuit;
    // Classify gates by dominant coefficient
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
        "c — bias": cCount,
        "a — first": aCount,
        "b — second": bCount,
        "p — interaction": pCount,
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
      { key: "c", avg: totals.const / count,   fill: COLORS.const },
      { key: "a", avg: totals.first / count,   fill: COLORS.first },
      { key: "b", avg: totals.second / count,  fill: COLORS.second },
      { key: "p", avg: totals.product / count,  fill: COLORS.product },
    ];

    // For large circuits, compute summary distribution instead of per-layer chart
    const totalGates = totalC + totalA + totalB + totalP;
    const _summaryData = [
      { key: "c", pct: (totalC / totalGates * 100), fill: COLORS.const },
      { key: "a", pct: (totalA / totalGates * 100), fill: COLORS.first },
      { key: "b", pct: (totalB / totalGates * 100), fill: COLORS.second },
      { key: "p", pct: (totalP / totalGates * 100), fill: COLORS.product },
    ];

    perfEnd('gatestats-compute');
    return {
      layerData: _layerData,
      coeffData: _coeffData,
      summaryData: _summaryData,
      showPerLayerChart: d <= MAX_LAYER_BARS,
    };
  }, [circuit]);

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
          {showPerLayerChart ? (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={layerData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F3F5" />
                <XAxis
                  dataKey="layer"
                  tick={{ fontSize: 10, fill: "#9CA3AF", fontFamily: "'IBM Plex Mono', monospace" }}
                  axisLine={{ stroke: "#E0E0E0" }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "#9CA3AF" }}
                  axisLine={{ stroke: "#E0E0E0" }}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: "#fff",
                    border: "1px solid #E0E0E0",
                    borderRadius: 8,
                    fontSize: 11,
                  }}
                />
                {["c — bias", "a — first", "b — second", "p — interaction"].map((key, ki) => {
                  const fills = [COLORS.const, COLORS.first, COLORS.second, COLORS.product];
                  return (
                    <Bar key={ki} dataKey={key} fill={fills[ki]} stackId="a">
                      {activeLayer !== undefined && activeLayer !== null &&
                        layerData.map((_, idx) => (
                          <Cell key={idx} fillOpacity={idx === activeLayer ? 1 : 0.3} />
                        ))}
                    </Bar>
                  );
                })}
              </BarChart>
            </ResponsiveContainer>
          ) : (
            /* Compact summary for large circuits — just 4 bars, not 256 */
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={summaryData} margin={{ top: 4, right: 8, bottom: 32, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F3F5" />
                <XAxis
                  dataKey="key"
                  tick={<ColoredTick />}
                  axisLine={{ stroke: "#E0E0E0" }}
                  tickLine={false}
                  interval={0}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "#9CA3AF" }}
                  axisLine={{ stroke: "#E0E0E0" }}
                  tickLine={false}
                  tickFormatter={(v) => `${v.toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{
                    background: "#fff",
                    border: "1px solid #E0E0E0",
                    borderRadius: 8,
                    fontSize: 11,
                  }}
                  formatter={(value, name, props) => {
                    const meta = COEFF_META[props.payload.key];
                    return [`${value.toFixed(1)}%`, meta ? `${props.payload.key} — ${meta.label}` : "% gates"];
                  }}
                />
                <Bar dataKey="pct" radius={[3, 3, 0, 0]}>
                  {summaryData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div>
          <h3 className="subheading">Avg |Coefficient| Magnitude</h3>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={coeffData} margin={{ top: 4, right: 8, bottom: 32, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F3F5" />
              <XAxis
                dataKey="key"
                tick={<ColoredTick />}
                axisLine={{ stroke: "#E0E0E0" }}
                tickLine={false}
                interval={0}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "#9CA3AF" }}
                axisLine={{ stroke: "#E0E0E0" }}
                tickLine={false}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{
                  background: "#fff",
                  border: "1px solid #E0E0E0",
                  borderRadius: 8,
                  fontSize: 11,
                }}
                formatter={(value, name, props) => {
                  const meta = COEFF_META[props.payload.key];
                  return [value.toFixed(4), meta ? `${props.payload.key} — ${meta.label}` : "Avg |coeff|"];
                }}
              />
              <Bar dataKey="avg" radius={[3, 3, 0, 0]}>
                {coeffData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
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
