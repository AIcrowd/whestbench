/**
 * SignalHeatmap — SVG heatmap of wire means across layers, using Recharts-style responsive container.
 * Each cell is an SVG rect — never pixelated.
 */

export default function SignalHeatmap({ means, width: n, depth: d }) {
  if (!means || means.length === 0) return null;

  const maxCellW = 32;
  const maxCellH = 24;
  const cellW = Math.max(8, Math.min(maxCellW, Math.floor(500 / n)));
  const cellH = Math.max(8, Math.min(maxCellH, Math.floor(320 / d)));
  const padL = 28;
  const padT = 16;
  const padB = 20;
  const padR = 8;

  const svgW = padL + n * cellW + padR;
  const svgH = padT + d * cellH + padB;

  const meanColor = (val) => {
    const t = (val + 1) / 2;
    if (t < 0.5) {
      const p = t * 2;
      const r = Math.round(59 + p * (156 - 59));
      const g = Math.round(130 + p * (163 - 130));
      const b = Math.round(246 + p * (175 - 246));
      return `rgb(${r},${g},${b})`;
    } else {
      const p = (t - 0.5) * 2;
      const r = Math.round(156 + p * (240 - 156));
      const g = Math.round(163 + p * (82 - 163));
      const b = Math.round(175 + p * (77 - 175));
      return `rgb(${r},${g},${b})`;
    }
  };

  const cells = [];
  for (let layer = 0; layer < d && layer < means.length; layer++) {
    for (let wire = 0; wire < n; wire++) {
      const val = means[layer][wire] || 0;
      cells.push(
        <rect
          key={`${layer}-${wire}`}
          x={padL + wire * cellW}
          y={padT + layer * cellH}
          width={cellW - 1}
          height={cellH - 1}
          rx={2}
          fill={meanColor(val)}
        >
          <title>L{layer} w{wire}: {val.toFixed(4)}</title>
        </rect>
      );
    }
  }

  // Layer labels (Y axis)
  const layerLabels = [];
  const labelStep = d > 16 ? Math.ceil(d / 8) : 1;
  for (let l = 0; l < d; l += labelStep) {
    layerLabels.push(
      <text
        key={`yl-${l}`}
        x={padL - 4}
        y={padT + l * cellH + cellH / 2 + 4}
        fontSize={9}
        fill="#9CA3AF"
        textAnchor="end"
        fontFamily="'IBM Plex Mono', monospace"
      >
        {l}
      </text>
    );
  }

  // Wire labels (X axis)
  const wireLabels = [];
  const wireLabelStep = n > 16 ? Math.ceil(n / 8) : 1;
  for (let w = 0; w < n; w += wireLabelStep) {
    wireLabels.push(
      <text
        key={`xl-${w}`}
        x={padL + w * cellW + cellW / 2}
        y={padT + d * cellH + 14}
        fontSize={9}
        fill="#9CA3AF"
        textAnchor="middle"
        fontFamily="'IBM Plex Mono', monospace"
      >
        {w}
      </text>
    );
  }

  return (
    <div className="panel">
      <h2>Wire Means Heatmap</h2>
      <div style={{ overflowX: "auto" }}>
        <svg width={svgW} height={svgH} style={{ display: "block" }}>
          {cells}
          {layerLabels}
          {wireLabels}
          <text x={padL - 4} y={padT - 4} fontSize={9} fill="#9CA3AF" textAnchor="end">Layer</text>
          <text x={padL + n * cellW} y={padT + d * cellH + 14} fontSize={9} fill="#9CA3AF" textAnchor="end">Wire</text>
        </svg>
      </div>
      <div className="heatmap-legend">
        <span className="legend-item" style={{ color: "#334155" }}>◆ −1</span>
        <span className="legend-item" style={{ color: "#9CA3AF" }}>◆ 0</span>
        <span className="legend-item" style={{ color: "#F0524D" }}>◆ +1</span>
      </div>
    </div>
  );
}
