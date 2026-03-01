/**
 * CircuitGraphJoint — JointJS circuit graph with proper port-based wiring.
 *
 * Each gate has exactly 2 inputs (first, second) from the previous layer.
 * Gate output feeds into the next layer.
 *
 * JointJS ports:
 *   - Group 'in':  2 ports on the LEFT  (in1, in2) — the two input wires
 *   - Group 'out': 1 port on the RIGHT  (out)      — the output wire
 *
 * Links connect: source.port='out' → target.port='in1' or 'in2'
 *
 * Layout: strict grid — d columns × n rows, uniform gate size.
 */
import { dia, shapes } from "@joint/core";
import { useEffect, useRef, useState } from "react";
import {
    classifyGate,
    GATE_H,
    GATE_TYPES,
    GATE_W,
    meanToColor,
} from "./gateShapes";

/* ------------------------------------------------------------------ */
/*  Layout constants                                                   */
/* ------------------------------------------------------------------ */
const COL_GAP = 100; // horizontal gap between layer columns
const ROW_GAP = 12;  // vertical gap between wire rows
const PAD_X = 40;
const PAD_Y = 40;

/* Port group definitions for JointJS */
const PORT_GROUPS = {
  in: {
    position: { name: "left" },
    attrs: {
      portBody: {
        r: 3,
        fill: "#94A3B8",
        stroke: "#64748B",
        strokeWidth: 1,
        magnet: false,
      },
    },
    markup: [{ tagName: "circle", selector: "portBody" }],
  },
  out: {
    position: { name: "right" },
    attrs: {
      portBody: {
        r: 3,
        fill: "#94A3B8",
        stroke: "#64748B",
        strokeWidth: 1,
        magnet: false,
      },
    },
    markup: [{ tagName: "circle", selector: "portBody" }],
  },
};

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
export default function CircuitGraphJoint({ circuit, means, activeLayer }) {
  const canvasRef = useRef(null);
  const paperRef = useRef(null);
  const graphRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [zoomPct, setZoomPct] = useState(100);

  /* ---- build / rebuild the graph ---- */
  useEffect(() => {
    const el = canvasRef.current;
    if (!el || !circuit) return;
    el.innerHTML = "";

    // 1. Graph
    const graph = new dia.Graph({}, { cellNamespace: shapes });
    graphRef.current = graph;

    // 2. Paper — frozen + async per official React docs
    const paper = new dia.Paper({
      model: graph,
      background: { color: "#FCFCFC" },
      frozen: true,
      async: true,
      cellViewNamespace: shapes,
      width: 1,
      height: 1,
      gridSize: 1,
      interactive: false,
    });
    paperRef.current = paper;
    el.appendChild(paper.el);

    // 3. Create gate elements in strict grid
    const nodes = []; // nodes[layer][wire]
    for (let l = 0; l < circuit.d; l++) {
      nodes[l] = [];
      for (let w = 0; w < circuit.n; w++) {
        const info = classifyGate(circuit.gates[l], w);
        const style = GATE_TYPES[info.type];
        const mean = means?.[l]?.[w] ?? null;
        const fill = meanToColor(mean) || style.fill;

        const x = PAD_X + l * (GATE_W + COL_GAP);
        const y = PAD_Y + w * (GATE_H + ROW_GAP);

        // Use standard.Path for the custom SVG shape
        const node = new shapes.standard.Path({
          position: { x, y },
          size: { width: GATE_W, height: GATE_H },
          attrs: {
            body: {
              d: style.path,
              fill: fill,
              stroke: style.stroke,
              strokeWidth: 1.5,
            },
            label: {
              text: String(w),
              fontSize: 10,
              fontFamily: "'IBM Plex Mono', monospace",
              fill: style.textColor,
              textAnchor: "middle",
              textVerticalAnchor: "middle",
            },
          },
          ports: {
            groups: PORT_GROUPS,
            items: [
              { id: "in1", group: "in" },
              { id: "in2", group: "in" },
              { id: "out", group: "out" },
            ],
          },
        });

        // Store metadata for click-to-inspect
        node.set("gateData", {
          layerIndex: l,
          wireIndex: w,
          type: info.type,
          label: info.label,
          first: circuit.gates[l].first[w],
          second: circuit.gates[l].second[w],
          const: circuit.gates[l].const[w],
          firstCoeff: circuit.gates[l].firstCoeff[w],
          secondCoeff: circuit.gates[l].secondCoeff[w],
          productCoeff: circuit.gates[l].productCoeff[w],
          mean,
        });

        nodes[l][w] = node;
        graph.addCell(node);
      }
    }

    // 4. Links — connect output port → input ports
    for (let l = 1; l < circuit.d; l++) {
      for (let w = 0; w < circuit.n; w++) {
        const g = circuit.gates[l];
        const fw = g.first[w];  // first input wire index
        const sw = g.second[w]; // second input wire index

        if (!nodes[l - 1]?.[fw] || !nodes[l]?.[w]) continue;

        // First input: previous layer wire fw → this gate in1
        graph.addCell(
          new shapes.standard.Link({
            source: { id: nodes[l - 1][fw].id, port: "out" },
            target: { id: nodes[l][w].id, port: "in1" },
            attrs: {
              line: {
                stroke: "#94A3B8",
                strokeWidth: 1,
                targetMarker: { d: "" }, // no arrowhead
              },
            },
            connector: { name: "smooth" },
          })
        );

        // Second input: previous layer wire sw → this gate in2
        if (sw !== fw) {
          graph.addCell(
            new shapes.standard.Link({
              source: { id: nodes[l - 1][sw].id, port: "out" },
              target: { id: nodes[l][w].id, port: "in2" },
              attrs: {
                line: {
                  stroke: "#CBD5E1",
                  strokeWidth: 0.8,
                  strokeDasharray: "4,3",
                  targetMarker: { d: "" },
                },
              },
              connector: { name: "smooth" },
            })
          );
        } else {
          // Both inputs from same wire — connect to in2 as well (solid)
          graph.addCell(
            new shapes.standard.Link({
              source: { id: nodes[l - 1][fw].id, port: "out" },
              target: { id: nodes[l][w].id, port: "in2" },
              attrs: {
                line: {
                  stroke: "#94A3B8",
                  strokeWidth: 1,
                  targetMarker: { d: "" },
                },
              },
              connector: { name: "smooth" },
            })
          );
        }
      }
    }

    // 5. CRITICAL: Unfreeze to trigger rendering
    paper.unfreeze();

    // 6. Fit content after async render
    requestAnimationFrame(() => {
      if (!paperRef.current) return;
      const bbox = graph.getBBox();
      if (!bbox) return;
      paper.setDimensions(bbox.x + bbox.width + 60, bbox.y + bbox.height + 50);
      paper.transformToFitContent({ padding: 20, maxScale: 1.5 });
      setZoomPct(Math.round(paper.scale().sx * 100));
    });

    // 7. Click to inspect gate
    paper.on("element:pointerclick", (view) => {
      const d = view.model.get("gateData");
      if (d) setTooltip(d);
    });
    paper.on("blank:pointerclick", () => setTooltip(null));

    return () => {
      paper.remove();
      paperRef.current = null;
      graphRef.current = null;
    };
  }, [circuit, means]);

  /* ---- zoom via wheel ---- */
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      e.stopPropagation();
      const paper = paperRef.current;
      if (!paper) return;
      const f = e.deltaY > 0 ? 0.92 : 1.08;
      const s = paper.scale().sx;
      const ns = Math.max(0.15, Math.min(4, s * f));
      paper.scale(ns, ns);
      setZoomPct(Math.round(ns * 100));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  /* ---- layer dimming ---- */
  useEffect(() => {
    const g = graphRef.current;
    if (!g) return;
    g.getElements().forEach((el) => {
      const d = el.get("gateData");
      if (!d) return;
      const dim = activeLayer !== undefined && d.layerIndex !== activeLayer;
      el.attr("body/opacity", dim ? 0.12 : 1);
      el.attr("label/opacity", dim ? 0.12 : 1);
    });
    g.getLinks().forEach((lk) => {
      lk.attr("line/opacity", activeLayer !== undefined ? 0.06 : 1);
    });
  }, [activeLayer]);

  /* ---- render ---- */
  return (
    <div className="panel circuit-graph-joint" style={{ position: "relative" }}>
      <h2>
        Circuit Structure
        <span className="mode-badge">
          Graph Mode · {circuit.n}×{circuit.d} = {circuit.n * circuit.d} gates
        </span>
      </h2>

      {/* Legend */}
      <div className="gate-legend">
        <span className="legend-item">
          <svg width="18" height="14" viewBox="0 0 18 14">
            <path d="M0 0L9 0C18 0 18 14 9 14L0 14Z" fill="#FEE2E2" stroke="#EF4444" strokeWidth="1.5"/>
          </svg>
          AND
        </span>
        <span className="legend-item">
          <svg width="18" height="14" viewBox="0 0 18 14">
            <path d="M0 0L16 7L0 14Z" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="1.5"/>
          </svg>
          Linear
        </span>
        <span className="legend-item">
          <svg width="14" height="14" viewBox="0 0 14 14">
            <circle cx="7" cy="7" r="6" fill="#FEF3C7" stroke="#F59E0B" strokeWidth="1.5"/>
          </svg>
          Product
        </span>
        <span className="legend-item">
          <svg width="14" height="14" viewBox="0 0 14 14">
            <rect x="1" y="1" width="12" height="12" fill="#F3F4F6" stroke="#9CA3AF" strokeWidth="1.5"/>
          </svg>
          Constant
        </span>
        <span className="legend-wire">
          <svg width="24" height="6" viewBox="0 0 24 6"><line x1="0" y1="3" x2="24" y2="3" stroke="#94A3B8" strokeWidth="1.5"/></svg>
          1st input
        </span>
        <span className="legend-wire">
          <svg width="24" height="6" viewBox="0 0 24 6"><line x1="0" y1="3" x2="24" y2="3" stroke="#CBD5E1" strokeWidth="1" strokeDasharray="4,3"/></svg>
          2nd input
        </span>
      </div>

      {/* JointJS canvas */}
      <div
        ref={canvasRef}
        className="joint-container"
        style={{
          overflow: "hidden",
          minHeight: 350,
          maxHeight: 600,
          border: "1px solid #E5E7EB",
          borderRadius: 8,
          background: "#FCFCFC",
        }}
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="gate-tooltip"
          style={{
            position: "absolute",
            left: "50%",
            top: 90,
            transform: "translateX(-50%)",
            zIndex: 200,
          }}
        >
          <div className="tooltip-header">
            Layer {tooltip.layerIndex}, Wire {tooltip.wireIndex}
          </div>
          <div className="tooltip-body">
            <div className="tooltip-op">{tooltip.label}</div>
            <div className="tooltip-coeffs">
              c={tooltip.const.toFixed(3)}, a={tooltip.firstCoeff.toFixed(3)},
              b={tooltip.secondCoeff.toFixed(3)}, p={tooltip.productCoeff.toFixed(3)}
            </div>
            <div className="tooltip-inputs">
              inputs: wire[{tooltip.first}], wire[{tooltip.second}]
            </div>
            {tooltip.mean !== null && (
              <div className="tooltip-mean">E[wire] = {tooltip.mean.toFixed(4)}</div>
            )}
          </div>
        </div>
      )}

      <div className="zoom-indicator">
        Zoom: {zoomPct}% · Scroll to zoom · Click gates to inspect
      </div>
    </div>
  );
}
