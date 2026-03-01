/**
 * CircuitGraphJoint — Interactive circuit diagram using pure SVG.
 * Used for small circuits (n×d ≤ 4096).
 * Shows gates as styled shapes with labels, connection lines between layers.
 * Supports zoom/pan, click-to-inspect.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { classifyGate, meanToColor } from "./gateShapes";

const GATE_W = 54;
const GATE_H = 32;
const H_GAP = 90;
const V_GAP = 10;
const PAD_LEFT = 50;
const PAD_TOP = 30;

function GateNode({ gateInfo, x, y, mean, onClick }) {
  const fill = mean !== null ? meanToColor(mean) : "#F3F4F6";
  const r = gateInfo.shape === "circle" ? 14 : gateInfo.shape === "triangle" ? 2 : 4;

  return (
    <g
      transform={`translate(${x},${y})`}
      onClick={onClick}
      style={{ cursor: "pointer" }}
    >
      <rect
        width={GATE_W}
        height={GATE_H}
        rx={r}
        ry={r}
        fill={fill}
        stroke={gateInfo.color}
        strokeWidth={2}
      />
      <text
        x={GATE_W / 2}
        y={GATE_H / 2}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={8}
        fontFamily="'IBM Plex Mono', monospace"
        fill="#374151"
      >
        {gateInfo.label.length > 14 ? gateInfo.label.slice(0, 14) : gateInfo.label}
      </text>
    </g>
  );
}

function ConnectionLine({ x1, y1, x2, y2, dashed }) {
  // Route: horizontal from source center-right, then vertical, then horizontal to target center-left
  const midX = (x1 + x2) / 2;
  const d = `M${x1},${y1} C${midX},${y1} ${midX},${y2} ${x2},${y2}`;
  return (
    <path
      d={d}
      fill="none"
      stroke={dashed ? "#E2E8F0" : "#CBD5E1"}
      strokeWidth={dashed ? 0.8 : 1}
      strokeDasharray={dashed ? "3,2" : "none"}
    />
  );
}

export default function CircuitGraphJoint({ circuit, means, activeLayer }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [viewBox, setViewBox] = useState(null);
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, vx: 0, vy: 0 });

  const totalWidth = PAD_LEFT + circuit.d * (GATE_W + H_GAP) + 20;
  const totalHeight = PAD_TOP + circuit.n * (GATE_H + V_GAP) + 20;

  // Initial viewBox
  useEffect(() => {
    setViewBox({ x: 0, y: 0, w: totalWidth, h: totalHeight });
  }, [totalWidth, totalHeight]);

  // Build gate positions and links
  const { gates, links } = useMemo(() => {
    const gateList = [];
    const linkList = [];

    for (let l = 0; l < circuit.d; l++) {
      for (let w = 0; w < circuit.n; w++) {
        const gateInfo = classifyGate(circuit.gates[l], w);
        const x = PAD_LEFT + l * (GATE_W + H_GAP);
        const y = PAD_TOP + w * (GATE_H + V_GAP);
        const mean = means && means[l] ? means[l][w] : null;
        gateList.push({ l, w, x, y, gateInfo, mean });
      }
    }

    // Build connections
    for (let l = 1; l < circuit.d; l++) {
      for (let w = 0; w < circuit.n; w++) {
        const gate = circuit.gates[l];
        const firstWire = gate.first[w];
        const secondWire = gate.second[w];

        const srcX = PAD_LEFT + (l - 1) * (GATE_W + H_GAP) + GATE_W;
        const tgtX = PAD_LEFT + l * (GATE_W + H_GAP);

        // First input
        const srcY1 = PAD_TOP + firstWire * (GATE_H + V_GAP) + GATE_H / 2;
        const tgtY1 = PAD_TOP + w * (GATE_H + V_GAP) + GATE_H / 3;
        linkList.push({ x1: srcX, y1: srcY1, x2: tgtX, y2: tgtY1, dashed: false, l });

        // Second input
        const srcY2 = PAD_TOP + secondWire * (GATE_H + V_GAP) + GATE_H / 2;
        const tgtY2 = PAD_TOP + w * (GATE_H + V_GAP) + (2 * GATE_H) / 3;
        linkList.push({ x1: srcX, y1: srcY2, x2: tgtX, y2: tgtY2, dashed: true, l });
      }
    }

    return { gates: gateList, links: linkList };
  }, [circuit, means]);

  // Zoom handler
  const handleWheel = useCallback(
    (e) => {
      e.preventDefault();
      if (!viewBox) return;
      const factor = e.deltaY > 0 ? 1.1 : 0.9;
      const svg = svgRef.current;
      const rect = svg.getBoundingClientRect();
      const mx = ((e.clientX - rect.left) / rect.width) * viewBox.w + viewBox.x;
      const my = ((e.clientY - rect.top) / rect.height) * viewBox.h + viewBox.y;

      const newW = viewBox.w * factor;
      const newH = viewBox.h * factor;
      setViewBox({
        x: mx - (mx - viewBox.x) * factor,
        y: my - (my - viewBox.y) * factor,
        w: newW,
        h: newH,
      });
    },
    [viewBox]
  );

  // Pan handlers
  const handleMouseDown = useCallback(
    (e) => {
      if (e.button !== 0) return;
      setIsPanning(true);
      panStart.current = {
        x: e.clientX,
        y: e.clientY,
        vx: viewBox.x,
        vy: viewBox.y,
      };
    },
    [viewBox]
  );

  const handleMouseMove = useCallback(
    (e) => {
      if (!isPanning || !viewBox) return;
      const svg = svgRef.current;
      const rect = svg.getBoundingClientRect();
      const dx = ((e.clientX - panStart.current.x) / rect.width) * viewBox.w;
      const dy = ((e.clientY - panStart.current.y) / rect.height) * viewBox.h;
      setViewBox((v) => ({
        ...v,
        x: panStart.current.vx - dx,
        y: panStart.current.vy - dy,
      }));
    },
    [isPanning, viewBox]
  );

  const handleMouseUp = useCallback(() => setIsPanning(false), []);

  const handleGateClick = useCallback(
    (l, w, x, y) => {
      const gate = circuit.gates[l];
      const gateInfo = classifyGate(gate, w);
      setTooltip({
        x: x + GATE_W / 2,
        y,
        data: {
          layerIndex: l,
          wireIndex: w,
          type: gateInfo.type,
          label: gateInfo.label,
          first: gate.first[w],
          second: gate.second[w],
          const: gate.const[w],
          firstCoeff: gate.firstCoeff[w],
          secondCoeff: gate.secondCoeff[w],
          productCoeff: gate.productCoeff[w],
          mean: means && means[l] ? means[l][w] : null,
        },
      });
    },
    [circuit, means]
  );

  const zoomPercent = viewBox
    ? Math.round((totalWidth / viewBox.w) * 100)
    : 100;

  return (
    <div className="panel circuit-graph-joint" ref={containerRef}>
      <h2>
        Circuit Structure
        <span className="mode-badge">
          Graph Mode · {circuit.n}×{circuit.d} = {circuit.n * circuit.d} gates
        </span>
      </h2>
      <svg
        ref={svgRef}
        className="joint-container"
        viewBox={
          viewBox
            ? `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`
            : `0 0 ${totalWidth} ${totalHeight}`
        }
        preserveAspectRatio="xMidYMid meet"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={(e) => {
          if (e.target === svgRef.current) setTooltip(null);
        }}
        style={{ cursor: isPanning ? "grabbing" : "grab" }}
      >
        {/* Background */}
        <rect
          x={viewBox?.x || 0}
          y={viewBox?.y || 0}
          width={viewBox?.w || totalWidth}
          height={viewBox?.h || totalHeight}
          fill="#FFFFFF"
        />

        {/* Layer labels */}
        {Array.from({ length: circuit.d }, (_, l) => (
          <text
            key={`ll${l}`}
            x={PAD_LEFT + l * (GATE_W + H_GAP) + GATE_W / 2}
            y={PAD_TOP - 12}
            textAnchor="middle"
            fontSize={9}
            fill="#9CA3AF"
            fontFamily="'IBM Plex Mono', monospace"
          >
            L{l}
          </text>
        ))}

        {/* Connections (behind gates) */}
        <g opacity={activeLayer !== undefined ? 0.06 : 0.6}>
          {links.map((link, i) => (
            <ConnectionLine key={i} {...link} />
          ))}
        </g>

        {/* Gates */}
        {gates.map(({ l, w, x, y, gateInfo, mean }) => (
          <g
            key={`g${l}_${w}`}
            opacity={
              activeLayer !== undefined && l !== activeLayer ? 0.15 : 1
            }
          >
            <GateNode
              gateInfo={gateInfo}
              x={x}
              y={y}
              mean={mean}
              onClick={() => handleGateClick(l, w, x, y)}
            />
          </g>
        ))}
      </svg>

      {/* Tooltip overlay */}
      {tooltip && (
        <div
          className="gate-tooltip"
          style={{
            position: "absolute",
            left: "50%",
            top: 60,
            transform: "translateX(-50%)",
          }}
        >
          <div className="tooltip-header">
            Layer {tooltip.data.layerIndex}, Wire {tooltip.data.wireIndex}
          </div>
          <div className="tooltip-body">
            <div className="tooltip-op">{tooltip.data.label}</div>
            <div className="tooltip-coeffs">
              c={tooltip.data.const.toFixed(3)},
              a={tooltip.data.firstCoeff.toFixed(3)},
              b={tooltip.data.secondCoeff.toFixed(3)},
              p={tooltip.data.productCoeff.toFixed(3)}
            </div>
            <div className="tooltip-inputs">
              inputs: x[{tooltip.data.first}], y[{tooltip.data.second}]
            </div>
            {tooltip.data.mean !== null && (
              <div className="tooltip-mean">
                E[wire] = {tooltip.data.mean.toFixed(4)}
              </div>
            )}
          </div>
        </div>
      )}
      <div className="zoom-indicator">
        Zoom: {zoomPercent}% · Scroll to zoom, drag to pan, click gates to
        inspect
      </div>
    </div>
  );
}
