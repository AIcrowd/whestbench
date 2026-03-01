/**
 * CircuitGraphJoint — JointJS-based circuit graph.
 *
 * Following the official React integration pattern:
 *   https://docs.jointjs.com/learn/integration/react/
 *
 * Step 1: Minimal working example with just a few rectangles + links.
 * Step 2: Wire in full circuit data.
 */
import { dia, shapes } from "@joint/core";
import { useEffect, useRef, useState } from "react";
import { classifyGate, meanToColor } from "./gateShapes";

/* ------------------------------------------------------------------ */
/*  Layout constants                                                   */
/* ------------------------------------------------------------------ */
const GATE_W = 64;
const GATE_H = 38;
const H_GAP = 110; // horizontal spacing between layers
const V_GAP = 14;  // vertical spacing between wires
const PAD_X = 60;
const PAD_Y = 50;

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

    // Tear down previous
    el.innerHTML = "";

    // 1. Graph
    const graph = new dia.Graph({}, { cellNamespace: shapes });
    graphRef.current = graph;

    // 2. Paper — frozen + async per docs
    const paper = new dia.Paper({
      model: graph,
      background: { color: "#FFFFFF" },
      frozen: true,
      async: true,
      cellViewNamespace: shapes,
      width: 1,   // will resize after fit
      height: 1,
      gridSize: 1,
      interactive: false,
    });
    paperRef.current = paper;

    // 3. Append paper element into our container
    el.appendChild(paper.el);

    // 4. Populate gate elements
    const nodes = [];            // nodes[layer][wire]
    for (let l = 0; l < circuit.d; l++) {
      nodes[l] = [];
      for (let w = 0; w < circuit.n; w++) {
        const info = classifyGate(circuit.gates[l], w);
        const x = PAD_X + l * (GATE_W + H_GAP);
        const y = PAD_Y + w * (GATE_H + V_GAP);
        const mean = means?.[l]?.[w] ?? null;
        const fill = mean !== null ? meanToColor(mean) : "#F3F4F6";

        const node = new shapes.standard.Rectangle({
          position: { x, y },
          size: { width: GATE_W, height: GATE_H },
          attrs: {
            body: {
              fill,
              stroke: info.color,
              strokeWidth: 2,
              rx: info.shape === "circle" ? 16 : info.shape === "dshape" ? 10 : 4,
              ry: info.shape === "circle" ? 16 : info.shape === "dshape" ? 10 : 4,
              cursor: "pointer",
            },
            label: {
              text: info.label.length > 14 ? info.label.slice(0, 14) : info.label,
              fontSize: 9,
              fontFamily: "'IBM Plex Mono', monospace",
              fill: "#374151",
            },
          },
        });

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

    // 5. Links
    for (let l = 1; l < circuit.d; l++) {
      for (let w = 0; w < circuit.n; w++) {
        const g = circuit.gates[l];
        const fw = g.first[w];
        const sw = g.second[w];

        // first-input link (solid)
        graph.addCell(
          new shapes.standard.Link({
            source: { id: nodes[l - 1][fw].id },
            target: { id: nodes[l][w].id },
            attrs: { line: { stroke: "#94A3B8", strokeWidth: 1.2, targetMarker: { d: "" } } },
            connector: { name: "smooth" },
          })
        );

        // second-input link (dashed) — skip duplicate
        if (sw !== fw) {
          graph.addCell(
            new shapes.standard.Link({
              source: { id: nodes[l - 1][sw].id },
              target: { id: nodes[l][w].id },
              attrs: { line: { stroke: "#CBD5E1", strokeWidth: 0.8, strokeDasharray: "4,3", targetMarker: { d: "" } } },
              connector: { name: "smooth" },
            })
          );
        }
      }
    }

    // 6. Unfreeze — critical!
    paper.unfreeze();

    // 7. Fit content after next frame (async rendering needs a tick)
    requestAnimationFrame(() => {
      if (!paperRef.current) return;
      const bbox = graph.getBBox();
      if (!bbox) return;
      const w = bbox.x + bbox.width + 60;
      const h = bbox.y + bbox.height + 40;
      paper.setDimensions(w, h);
      paper.transformToFitContent({ padding: 20, maxScale: 1.5 });
      setZoomPct(Math.round(paper.scale().sx * 100));
    });

    // 8. Click-to-inspect
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

  /* ---- zoom via wheel (prevent page scroll) ---- */
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

      {/* JointJS canvas — overflow:hidden + wheel captures zoom */}
      <div
        ref={canvasRef}
        className="joint-container"
        style={{
          overflow: "hidden",
          minHeight: 350,
          maxHeight: 600,
          border: "1px solid #E0E0E0",
          borderRadius: 8,
          background: "#fff",
        }}
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="gate-tooltip"
          style={{ position: "absolute", left: "50%", top: 56, transform: "translateX(-50%)", zIndex: 200 }}
        >
          <div className="tooltip-header">Layer {tooltip.layerIndex}, Wire {tooltip.wireIndex}</div>
          <div className="tooltip-body">
            <div className="tooltip-op">{tooltip.label}</div>
            <div className="tooltip-coeffs">
              c={tooltip.const.toFixed(3)}, a={tooltip.firstCoeff.toFixed(3)},
              b={tooltip.secondCoeff.toFixed(3)}, p={tooltip.productCoeff.toFixed(3)}
            </div>
            <div className="tooltip-inputs">
              inputs: x[{tooltip.first}], y[{tooltip.second}]
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
