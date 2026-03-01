/**
 * WireStats — per-layer wire mean distribution with μ ± σ band chart.
 */
import {
    Area,
    CartesianGrid,
    ComposedChart,
    Line,
    ReferenceLine,
    ResponsiveContainer,
    Scatter,
    ScatterChart,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

export default function WireStats({ means, width: n, depth: d }) {
  if (!means || means.length === 0) return null;

  // Compute per-layer stats
  const layerStats = [];
  for (let l = 0; l < d && l < means.length; l++) {
    let sum = 0, sumSq = 0;
    let min = Infinity, max = -Infinity;
    for (let w = 0; w < n; w++) {
      const val = means[l][w] || 0;
      sum += val;
      sumSq += val * val;
      if (val < min) min = val;
      if (val > max) max = val;
    }
    const avg = sum / n;
    const variance = sumSq / n - avg * avg;
    const std = Math.sqrt(Math.max(0, variance));
    layerStats.push({
      layer: l,
      mean: avg,
      std,
      upper: Math.min(1, avg + std),
      lower: Math.max(-1, avg - std),
      band: [Math.max(-1, avg - std), Math.min(1, avg + std)],
      min,
      max,
    });
  }

  // Scatter data: every wire mean as a point
  const scatterData = [];
  for (let l = 0; l < d && l < means.length; l++) {
    for (let w = 0; w < n; w++) {
      scatterData.push({ layer: l, mean: means[l][w] || 0 });
    }
  }

  return (
    <div className="panel">
      <h2>Wire Mean Distribution</h2>

      {/* Mean ± σ band chart */}
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={layerStats} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F3F5" />
          <XAxis
            dataKey="layer"
            tick={{ fontSize: 10, fill: "#9CA3AF", fontFamily: "'IBM Plex Mono', monospace" }}
            axisLine={{ stroke: "#E0E0E0" }}
            tickLine={false}
            label={{ value: "Layer", position: "insideBottom", offset: -2, style: { fontSize: 10, fill: "#9CA3AF" } }}
          />
          <YAxis
            domain={[-1, 1]}
            tick={{ fontSize: 10, fill: "#9CA3AF", fontFamily: "'IBM Plex Mono', monospace" }}
            axisLine={{ stroke: "#E0E0E0" }}
            tickLine={false}
            label={{ value: "E[wire]", angle: -90, position: "insideLeft", style: { fontSize: 10, fill: "#9CA3AF" } }}
          />
          <ReferenceLine y={0} stroke="#E0E0E0" strokeDasharray="4 4" />
          <Tooltip
            contentStyle={{
              background: "#fff",
              border: "1px solid #E0E0E0",
              borderRadius: 8,
              fontSize: 11,
              fontFamily: "'IBM Plex Mono', monospace",
            }}
            formatter={(value, name) => {
              if (name === "band") return null;
              if (Array.isArray(value)) return [`[${value[0].toFixed(3)}, ${value[1].toFixed(3)}]`, "μ ± σ"];
              return [typeof value === "number" ? value.toFixed(4) : value, name];
            }}
          />

          {/* σ band */}
          <Area
            type="monotone"
            dataKey="band"
            fill="#3B82F6"
            fillOpacity={0.12}
            stroke="none"
            name="± σ"
          />

          {/* Mean line */}
          <Line
            type="monotone"
            dataKey="mean"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={{ r: 4, fill: "#3B82F6", stroke: "#fff", strokeWidth: 2 }}
            name="μ (mean)"
          />

          {/* Min/Max as thin dashed lines */}
          <Line
            type="monotone"
            dataKey="max"
            stroke="#F0524D"
            strokeWidth={1}
            strokeDasharray="3 3"
            dot={false}
            name="max"
          />
          <Line
            type="monotone"
            dataKey="min"
            stroke="#3B82F6"
            strokeWidth={1}
            strokeDasharray="3 3"
            dot={false}
            name="min"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Individual wire scatter overlay */}
      <h3 className="subheading" style={{ marginTop: 12 }}>Individual Wire Means</h3>
      <ResponsiveContainer width="100%" height={140}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F3F5" />
          <XAxis
            type="number"
            dataKey="layer"
            domain={[0, d - 1]}
            tick={{ fontSize: 10, fill: "#9CA3AF", fontFamily: "'IBM Plex Mono', monospace" }}
            axisLine={{ stroke: "#E0E0E0" }}
            tickLine={false}
          />
          <YAxis
            type="number"
            dataKey="mean"
            domain={[-1, 1]}
            tick={{ fontSize: 10, fill: "#9CA3AF", fontFamily: "'IBM Plex Mono', monospace" }}
            axisLine={{ stroke: "#E0E0E0" }}
            tickLine={false}
          />
          <ReferenceLine y={0} stroke="#E0E0E0" strokeDasharray="4 4" />
          <Tooltip
            contentStyle={{
              background: "#fff",
              border: "1px solid #E0E0E0",
              borderRadius: 8,
              fontSize: 11,
              fontFamily: "'IBM Plex Mono', monospace",
            }}
            formatter={(value) => [value.toFixed(4), undefined]}
          />
          <Scatter data={scatterData} fill="#3B82F6" fillOpacity={0.4} r={2.5} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
