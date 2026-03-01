/**
 * CircuitGraphJoint — JointJS circuit graph with uniform gates and
 * orthogonal (circuit-style) wire routing.
 *
 * Each gate has exactly 2 inputs (first, second) from the previous layer.
 * All gates are identical rectangles; type shown by border color.
 * Wires use JointJS manhattan router for right-angle circuit-style routing.
 *
 * JointJS ports:
 *   - Group 'in':  2 ports on LEFT  (in1, in2)
 *   - Group 'out': 1 port on RIGHT  (out)
 */
import { dia, shapes } from "@joint/core";
import { useEffect, useRef, useState } from "react";
import {
    classifyGate,
    GATE_H,
    GATE_W,
    gateColor,
    meanToColor,
} from "./gateShapes";

/* ------------------------------------------------------------------ */
/*  Layout                                                             */
/* ------------------------------------------------------------------ */
const COL_GAP = 80;  // space between layers (for wire routing)
const ROW_GAP = 10;  // space between wires
const PAD_X = 30;
const PAD_Y = 30;

/* JointJS port group definitions */
const PORT_GROUPS = {
  in: {
    position: { name: "left" },
    attrs: {
      portBody: {
        r: 2.5,
        fill: "#94A3B8",
        stroke: "none",
        magnet: false,
      },
    },
    markup: [{ tagName: "circle", selector: "portBody" }],
  },
  out: {
    position: { name: "right" },
    attrs: {
      portBody: {
        r: 2.5,
        fill: "#94A3B8",
        stroke: "none",
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

  useEffect(() => {
    const el = canvasRef.current;
    if (!el || !circuit) return;
    el.innerHTML = "";

    const graph = new dia.Graph({}, { cellNamespace: shapes });
    graphRef.current = graph;

    const paper = new dia.Paper({
      model: graph,
      background: { color: "#FCFCFC" },
      frozen: true,
      async: true,
      cellViewNamespace: shapes,
      width: 1,
      height: 1,
      gridSize: 10,
      interactive: false,
    });
    paperRef.current = paper;
    el.appendChild(paper.el);

    // --- Create gate elements (uniform rectangles) ---
    const nodes = [];
    for (let l = 0; l < circuit.d; l++) {
      nodes[l] = [];
      for (let w = 0; w < circuit.n; w++) {
        const info = classifyGate(circuit.gates[l], w);
        const colors = gateColor(info.type);
        const mean = means?.[l]?.[w] ?? null;
        const fill = meanToColor(mean) || "#FFFFFF";

        const x = PAD_X + l * (GATE_W + COL_GAP);
        const y = PAD_Y + w * (GATE_H + ROW_GAP);

        const node = new shapes.standard.Rectangle({
          position: { x, y },
          size: { width: GATE_W, height: GATE_H },
          attrs: {
            body: {
              fill,
              stroke: colors.stroke,
              strokeWidth: 1.5,
              rx: 3,
              ry: 3,
            },
            label: {
              text: String(w),
              fontSize: 10,
              fontFamily: "'IBM Plex Mono', monospace",
              fill: colors.text,
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

        node.set("gateData", {
          layerIndex: l,
          wireIndex: w,
          type: info.type,
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

    // --- Create links with manhattan routing ---
    for (let l = 1; l < circuit.d; l++) {
      for (let w = 0; w < circuit.n; w++) {
        const g = circuit.gates[l];
        const fw = g.first[w];
        const sw = g.second[w];

        if (!nodes[l - 1]?.[fw] || !nodes[l]?.[w]) continue;

        // First input
        graph.addCell(
          new shapes.standard.Link({
            source: { id: nodes[l - 1][fw].id, port: "out" },
            target: { id: nodes[l][w].id, port: "in1" },
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

        // Second input
        graph.addCell(
          new shapes.standard.Link({
            source: { id: nodes[l - 1][sw].id, port: "out" },
            target: { id: nodes[l][w].id, port: "in2" },
            attrs: {
              line: {
                stroke: fw === sw ? "#94A3B8" : "#CBD5E1",
                strokeWidth: fw === sw ? 1 : 0.8,
                strokeDasharray: fw === sw ? "" : "4,3",
                targetMarker: { d: "" },
              },
            },
            connector: { name: "smooth" },
          })
        );
      }
    }

    // Unfreeze
    paper.unfreeze();

    // Fit content to fill available container width
    requestAnimationFrame(() => {
      if (!paperRef.current || !canvasRef.current) return;
      const bbox = graph.getBBox();
      if (!bbox) return;
      const containerW = canvasRef.current.clientWidth;
      const contentH = bbox.y + bbox.height + 40;
      paper.setDimensions(containerW, contentH);
      paper.transformToFitContent({ padding: 20 });
      setZoomPct(Math.round(paper.scale().sx * 100));
    });

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

  /* ---- zoom ---- */
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

  return (
    <div className="panel circuit-graph-joint" style={{ position: "relative" }}>
      <h2>
        Circuit Structure
        <span className="mode-badge">
          Graph Mode · {circuit.n}×{circuit.d} = {circuit.n * circuit.d} gates
        </span>
      </h2>

      {/* Legend — minimal: just wire types + color meaning */}
      <div className="gate-legend">
        <span className="legend-item" style={{ color: "#EF4444" }}>■ AND</span>
        <span className="legend-item" style={{ color: "#3B82F6" }}>■ Linear</span>
        <span className="legend-item" style={{ color: "#F59E0B" }}>■ Product</span>
        <span className="legend-item" style={{ color: "#9CA3AF" }}>■ Constant</span>
        <span className="legend-wire">
          <svg width="24" height="6" viewBox="0 0 24 6"><line x1="0" y1="3" x2="24" y2="3" stroke="#94A3B8" strokeWidth="1.5"/></svg>
          1st
        </span>
        <span className="legend-wire">
          <svg width="24" height="6" viewBox="0 0 24 6"><line x1="0" y1="3" x2="24" y2="3" stroke="#CBD5E1" strokeWidth="1" strokeDasharray="4,3"/></svg>
          2nd
        </span>
      </div>

      <div
        ref={canvasRef}
        className="joint-container"
        style={{
          overflow: "hidden",
          minHeight: 350,
          border: "1px solid #E5E7EB",
          borderRadius: 8,
          width: "100%",
          background: "#FCFCFC",
        }}
      />

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
            Layer {tooltip.layerIndex}, Wire {tooltip.wireIndex} ({tooltip.type})
          </div>
          <div className="tooltip-body">
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
