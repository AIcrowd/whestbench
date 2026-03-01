/**
 * GateStats — visualizes the distribution of gate coefficients across the circuit.
 * Shows what kinds of operations dominate: constants, linear terms, or products.
 */
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

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
      constant: constants,
      linear,
      product: products,
      mixed,
    });
  }

  // Overall coefficient magnitude stats
  let totalConst = 0, totalFirst = 0, totalSecond = 0, totalProduct = 0;
  let count = 0;
  for (let l = 0; l < d; l++) {
    const layer = gates[l];
    for (let i = 0; i < n; i++) {
      totalConst += Math.abs(layer.const[i]);
      totalFirst += Math.abs(layer.firstCoeff[i]);
      totalSecond += Math.abs(layer.secondCoeff[i]);
      totalProduct += Math.abs(layer.productCoeff[i]);
      count++;
    }
  }

  const coeffData = [
    { name: "const", avg: totalConst / count },
    { name: "first", avg: totalFirst / count },
    { name: "second", avg: totalSecond / count },
    { name: "product", avg: totalProduct / count },
  ];

  return (
    <div className="panel">
      <h2>Gate Structure Analysis</h2>

      <div className="gate-stats-grid">
        <div>
          <h3 className="subheading">Dominant Operation per Layer</h3>
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
              <Bar dataKey="constant" name="Constant" fill="#D1D5DB" stackId="a" />
              <Bar dataKey="linear" name="Linear" fill="#94A3B8" stackId="a" />
              <Bar dataKey="product" name="Product" fill="#F0524D" stackId="a" />
              <Bar dataKey="mixed" name="Mixed" fill="#F7A09D" stackId="a" />
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
              <Bar dataKey="avg" name="Avg |coeff|" fill="#F0524D" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <p className="panel-desc">
        Each gate computes: <code>c + a·x + b·y + p·x·y</code> where x,y ∈ {"{"}−1,+1{"}"}.
        Product-heavy circuits are harder to estimate.
      </p>
    </div>
  );
}
