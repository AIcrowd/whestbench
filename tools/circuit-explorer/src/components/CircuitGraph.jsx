
import { describeGate } from "../circuit";

/**
 * SVG-based circuit graph visualization.
 * Shows wires as horizontal lanes, gates as nodes, and connections as curved paths.
 * Color encodes wire mean at each layer.
 */
export default function CircuitGraph({ circuit, means, activeLayer }) {
  const { n, d, gates } = circuit;

  // Layout constants
  const wireSpacing = 28;
  const layerSpacing = 120;
  const nodeRadius = 10;
  const padX = 60;
  const padY = 40;
  const width = padX * 2 + (d + 1) * layerSpacing;
  const height = padY * 2 + n * wireSpacing;

  // Color mapping: mean ∈ [-1, 1]
  // -1 → blue (#3B82F6), 0 → gray (#9CA3AF), +1 → coral (#F0524D)
  const meanColor = (val) => {
    if (val === undefined || isNaN(val)) return "#9CA3AF";
    const t = (val + 1) / 2; // 0..1
    if (t < 0.5) {
      // blue → gray
      const p = t * 2;
      const r = Math.round(59 + p * (156 - 59));
      const g = Math.round(130 + p * (163 - 130));
      const b = Math.round(246 + p * (175 - 246));
      return `rgb(${r},${g},${b})`;
    } else {
      // gray → coral
      const p = (t - 0.5) * 2;
      const r = Math.round(156 + p * (240 - 156));
      const g = Math.round(163 + p * (82 - 163));
      const b = Math.round(175 + p * (77 - 175));
      return `rgb(${r},${g},${b})`;
    }
  };

  // Wire positions
  const wireY = (i) => padY + i * wireSpacing + wireSpacing / 2;
  const layerX = (l) => padX + (l + 1) * layerSpacing;
  const inputX = padX;

  // Get mean for a wire at a given layer (layer -1 = inputs, mean=0)
  const getMean = (layerIdx, wireIdx) => {
    if (!means || layerIdx < 0) return 0;
    if (layerIdx >= means.length) return 0;
    return means[layerIdx][wireIdx] || 0;
  };

  const elements = (() => {
    const els = [];

    // Input labels
    for (let i = 0; i < n; i++) {
      els.push(
        <g key={`input-${i}`}>
          <circle
            cx={inputX}
            cy={wireY(i)}
            r={nodeRadius * 0.7}
            fill={meanColor(0)}
            stroke="#E0E0E0"
            strokeWidth={1}
          />
          <text
            x={inputX - 20}
            y={wireY(i) + 4}
            fontSize={10}
            fill="#9CA3AF"
            textAnchor="end"
            fontFamily="'IBM Plex Mono', monospace"
          >
            x{i}
          </text>
        </g>
      );
    }

    // For each layer, draw connections and gate nodes
    for (let l = 0; l < d; l++) {
      const layer = gates[l];
      const isActive = activeLayer === undefined || l <= activeLayer;
      const opacity = isActive ? 1 : 0.2;
      const lx = layerX(l);
      const prevX = l === 0 ? inputX : layerX(l - 1);

      for (let i = 0; i < n; i++) {
        const y = wireY(i);
        const y1 = wireY(layer.first[i]);
        const y2 = wireY(layer.second[i]);
        const wireMean = getMean(l, i);

        // Connection from first input
        els.push(
          <path
            key={`conn1-${l}-${i}`}
            d={`M${prevX},${y1} C${(prevX + lx) / 2},${y1} ${(prevX + lx) / 2},${y} ${lx},${y}`}
            fill="none"
            stroke={meanColor(getMean(l - 1, layer.first[i]))}
            strokeWidth={1.2}
            opacity={opacity * 0.5}
          />
        );

        // Connection from second input
        els.push(
          <path
            key={`conn2-${l}-${i}`}
            d={`M${prevX},${y2} C${(prevX + lx) / 2},${y2} ${(prevX + lx) / 2},${y} ${lx},${y}`}
            fill="none"
            stroke={meanColor(getMean(l - 1, layer.second[i]))}
            strokeWidth={1.2}
            opacity={opacity * 0.3}
            strokeDasharray="3,3"
          />
        );

        // Gate node
        els.push(
          <g key={`gate-${l}-${i}`} opacity={opacity}>
            <circle
              cx={lx}
              cy={y}
              r={nodeRadius}
              fill={meanColor(wireMean)}
              stroke={l === activeLayer ? "#F0524D" : "#E0E0E0"}
              strokeWidth={l === activeLayer ? 2 : 1}
            />
            <title>
              Layer {l}, Wire {i}: {describeGate(layer, i)} → mean ≈{" "}
              {wireMean.toFixed(3)}
            </title>
          </g>
        );
      }

      // Layer label
      els.push(
        <text
          key={`label-${l}`}
          x={lx}
          y={height - 8}
          fontSize={10}
          fill="#9CA3AF"
          textAnchor="middle"
          fontFamily="'IBM Plex Mono', monospace"
          opacity={opacity}
        >
          L{l}
        </text>
      );
    }

    return els;
  })();

  return (
    <div className="circuit-graph">
      <h2>Circuit Structure</h2>
      <div className="circuit-graph-scroll">
        <svg width={width} height={height} className="circuit-svg">
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {elements}
        </svg>
      </div>
      <div className="color-legend">
        <span style={{ color: meanColor(-1) }}>◆ mean = -1</span>
        <span style={{ color: meanColor(0) }}>◆ mean ≈ 0</span>
        <span style={{ color: meanColor(1) }}>◆ mean = +1</span>
      </div>
    </div>
  );
}
