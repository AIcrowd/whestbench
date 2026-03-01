/**
 * GateStats — visualizes the distribution of gate coefficients across the circuit.
 * Shows what kinds of operations dominate: constants, linear terms, or products.
 *
 * Gate formula: out = c + a·x + b·y + p·x·y
 *   c (const)   — fixed bias, always the same regardless of inputs
 *   a (first)   — weight on the first input wire x
 *   b (second)  — weight on the second input wire y
 *   p (product)  — weight on the interaction x·y (non-linear)
 */
import {
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    Legend,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

/* ── Strict palette from circuit gate colors (gateShapes.js) ── */
const COLORS = {
  const:   "#D1D5DB", // constant gate border
  first:   "#94A3B8", // linear gate border (slate)
  second:  "#334155", // dark slate (mean = -1 anchor)
  product: "#F0524D", // coral (product / and gate)
  mixed:   "#F7A09D", // coral-light tint
};

export default function GateStats({ circuit }) {
  if (!circuit) return null;
  const { n, d, gates } = circuit;

  // Classify gates by dominant coefficient
  const layerData = [];
  for (let l = 0; l < d; l++) {
    const layer = gates[l];
    let constants = 0;     // |const| > others
    let linear = 0;        // |first| + |second| > |const| + |product|
    let products = 0;      // |product| > others
    let mixed = 0;

    for (let i = 0; i < n; i++) {
      const c = Math.abs(layer.const[i]);
      const f = Math.abs(layer.firstCoeff[i]);
      const s = Math.abs(layer.secondCoeff[i]);
      const p = Math.abs(layer.productCoeff[i]);

      if (p >= c && p >= f && p >= s) products++;
      else if (c >= f && c >= s && c >= p) constants++;
      else if (f + s > c + p) linear++;
      else mixed++;
    }

    layerData.push({
      layer: `L${l}`,
      Constant: constants,
      Linear: linear,
      Product: products,
      Mixed: mixed,
    });
  }

  // Overall coefficient magnitude stats — use same color per coeff type
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

  const coeffData = [
    { name: "c (const)",   avg: totals.const / count,   fill: COLORS.const },
    { name: "a (first)",   avg: totals.first / count,   fill: COLORS.first },
    { name: "b (second)",  avg: totals.second / count,  fill: COLORS.second },
    { name: "p (product)", avg: totals.product / count,  fill: COLORS.product },
  ];

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
          <h3 className="subheading">Dominant Coefficient per Layer</h3>
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
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Bar dataKey="Constant" fill={COLORS.const} stackId="a" />
              <Bar dataKey="Linear" fill={COLORS.first} stackId="a" />
              <Bar dataKey="Product" fill={COLORS.product} stackId="a" />
              <Bar dataKey="Mixed" fill={COLORS.mixed} stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <h3 className="subheading">Avg |Coefficient| Magnitude</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={coeffData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F3F5" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10, fill: "#9CA3AF", fontFamily: "'IBM Plex Mono', monospace" }}
                axisLine={{ stroke: "#E0E0E0" }}
                tickLine={false}
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
                formatter={(value) => [value.toFixed(4), "Avg |coeff|"]}
              />
              {/* Each bar gets its own color from the data */}
              <Bar dataKey="avg" name="Avg |coeff|" radius={[3, 3, 0, 0]}>
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
