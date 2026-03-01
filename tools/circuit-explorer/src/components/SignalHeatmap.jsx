/**
 * SignalHeatmap — SVG heatmap of wire means across layers.
 * Uses container-measured width to fill the panel while keeping text readable.
 */
import { useEffect, useRef, useState } from "react";
import { meanToColor } from "./gateShapes";
export default function SignalHeatmap({ means, width: n, depth: d, source }) {
  const containerRef = useRef(null);
  const [containerW, setContainerW] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerW(entry.contentRect.width);
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  if (!means || means.length === 0) return null;

  const padL = 32;
  const padT = 8;
  const padB = 18;
  const padR = 4;

  // Compute cell sizes from available width
  const availableW = Math.max(100, containerW - padL - padR);
  const cellW = Math.max(8, Math.floor(availableW / n));
  const cellH = Math.max(8, Math.min(28, Math.floor(280 / d)));

  const svgW = padL + n * cellW + padR;
  const svgH = padT + d * cellH + padB;

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
          fill={meanToColor(val)}
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
        y={padT + l * cellH + cellH / 2 + 3}
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
        y={padT + d * cellH + 13}
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
      <h2>
        Wire Means Heatmap
        {source && <span className="source-badge">{source}</span>}
      </h2>
      <div ref={containerRef} style={{ width: "100%", overflowX: "auto" }}>
        {containerW > 0 && (
          <svg width={svgW} height={svgH} style={{ display: "block" }}>
            {cells}
            {layerLabels}
            {wireLabels}
            <text x={padL - 4} y={padT - 1} fontSize={9} fill="#9CA3AF" textAnchor="end">Layer</text>
            <text x={padL + n * cellW} y={padT + d * cellH + 13} fontSize={9} fill="#9CA3AF" textAnchor="end">Wire</text>
          </svg>
        )}
      </div>
      <div className="heatmap-legend">
        <span className="legend-item" style={{ color: "#334155" }}>◆ −1</span>
        <span className="legend-item" style={{ color: "#9CA3AF" }}>◆ 0</span>
        <span className="legend-item" style={{ color: "#F0524D" }}>◆ +1</span>
      </div>
    </div>
  );
}
