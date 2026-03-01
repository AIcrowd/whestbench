/**
 * EstimatorComparison — per-layer MSE comparison using Recharts SVG.
 * Shows bar chart of estimation error per layer for each estimator.
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

export default function EstimatorComparison({
  groundTruth,
  samplingEstimates,
  meanPropEstimates,
  depth,
}) {
  if (!groundTruth) return null;

  const hasSampling = !!samplingEstimates;
  const hasMeanProp = !!meanPropEstimates;

  const layers = Math.min(
    depth,
    groundTruth.length,
    hasSampling ? samplingEstimates.length : Infinity,
    hasMeanProp ? meanPropEstimates.length : Infinity
  );

  const data = [];
  for (let l = 0; l < layers; l++) {
    const n = groundTruth[l].length;
    const row = { layer: `L${l}` };

    if (hasSampling) {
      let mse = 0;
      for (let i = 0; i < n; i++) {
        mse += (samplingEstimates[l][i] - groundTruth[l][i]) ** 2;
      }
      row.sampling = mse / n;
    }

    if (hasMeanProp) {
      let mse = 0;
      for (let i = 0; i < n; i++) {
        mse += (meanPropEstimates[l][i] - groundTruth[l][i]) ** 2;
      }
      row.meanProp = mse / n;
    }

    data.push(row);
  }

  return (
    <div className="panel">
      <h2>Estimation Error (MSE per Layer)</h2>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F3F5" />
          <XAxis
            dataKey="layer"
            tick={{ fontSize: 11, fill: "#9CA3AF", fontFamily: "'IBM Plex Mono', monospace" }}
            axisLine={{ stroke: "#E0E0E0" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#9CA3AF", fontFamily: "'IBM Plex Mono', monospace" }}
            axisLine={{ stroke: "#E0E0E0" }}
            tickLine={false}
            tickFormatter={(v) => v.toFixed(3)}
            label={{
              value: "MSE",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 11, fill: "#9CA3AF" },
            }}
          />
          <Tooltip
            contentStyle={{
              background: "#fff",
              border: "1px solid #E0E0E0",
              borderRadius: 8,
              fontSize: 12,
              fontFamily: "'IBM Plex Mono', monospace",
            }}
            formatter={(value) => [value.toFixed(5), undefined]}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, fontFamily: "'DM Sans', sans-serif" }}
          />
          {hasSampling && (
            <Bar dataKey="sampling" name="Sampling" fill="#F0524D" radius={[3, 3, 0, 0]} />
          )}
          {hasMeanProp && (
            <Bar dataKey="meanProp" name="Mean Prop" fill="#94A3B8" radius={[3, 3, 0, 0]} />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
