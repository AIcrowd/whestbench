/**
 * CircuitGraphJoint — JointJS circuit graph with custom gate shapes.
 *
 * Uses custom shapes defined in gateShapes.js:
 *   AND:      D-shape (IEEE AND gate notation)
 *   Linear:   Triangle (buffer/amplifier)
 *   Product:  Circle with × cross (multiplier)
 *   Constant: Square with DC line (source)
 *
 * Official React integration: https://docs.jointjs.com/learn/integration/react/
 */
import { dia, shapes } from "@joint/core";
import { useEffect, useRef, useState } from "react";
import { classifyGate, GATE_CONSTRUCTORS, meanToColor } from "./gateShapes";

/* ------------------------------------------------------------------ */
/*  Layout constants                                                   */
/* ------------------------------------------------------------------ */
const H_GAP = 110; // horizontal spacing between layers
const V_GAP = 14;  // vertical spacing between wires
const PAD_X = 60;
const PAD_Y = 50;

/* ------------------------------------------------------------------ */
/*  Build custom cellNamespace merging standard + circuit shapes       */
/* ------------------------------------------------------------------ */
const cellNamespace = {
  ...shapes,
  circuit: {
    GateAND: GATE_CONSTRUCTORS.dshape,
    GateLinear: GATE_CONSTRUCTORS.triangle,
    GateProduct: GATE_CONSTRUCTORS.circle,
    GateConstant: GATE_CONSTRUCTORS.square,
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

    // Tear down previous
    el.innerHTML = "";

    // 1. Graph with merged namespace
    const graph = new dia.Graph({}, { cellNamespace });
    graphRef.current = graph;

    // 2. Paper — frozen + async per docs
    const paper = new dia.Paper({
      model: graph,
      background: { color: "#FCFCFC" },
      frozen: true,
      async: true,
      cellViewNamespace: cellNamespace,
      width: 1,
      height: 1,
      gridSize: 1,
      interactive: false,
    });
    paperRef.current = paper;

    // 3. Append paper element into our container
    el.appendChild(paper.el);

    // 4. Create gate elements using custom shapes
    const nodes = [];
    for (let l = 0; l < circuit.d; l++) {
      nodes[l] = [];
      for (let w = 0; w < circuit.n; w++) {
        const info = classifyGate(circuit.gates[l], w);
        const Constructor = GATE_CONSTRUCTORS[info.shape];
        if (!Constructor) continue;

        const defaultSize = Constructor.prototype.defaults.size || { width: 64, height: 38 };
        const gateW = defaultSize.width;
        const gateH = defaultSize.height;

        const x = PAD_X + l * (gateW + H_GAP);
        const y = PAD_Y + w * (gateH + V_GAP);
        const mean = means?.[l]?.[w] ?? null;
        const fill = mean !== null ? meanToColor(mean) : undefined; // only override if we have data

        const node = new Constructor({
          position: { x, y },
        });

        // Set label — just wire index; shape conveys type, details on click
        node.attr("label/text", String(w));

        // Override body fill if we have mean data
        if (fill) {
          node.attr("body/fill", fill);
        }

        // Store metadata for inspection
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

    // 5. Links — smooth connectors
    for (let l = 1; l < circuit.d; l++) {
      for (let w = 0; w < circuit.n; w++) {
        const g = circuit.gates[l];
        const fw = g.first[w];
        const sw = g.second[w];

        // First input connection (solid)
        if (nodes[l - 1]?.[fw] && nodes[l]?.[w]) {
          graph.addCell(
            new shapes.standard.Link({
              source: { id: nodes[l - 1][fw].id },
              target: { id: nodes[l][w].id },
              attrs: {
                line: {
                  stroke: "#94A3B8",
                  strokeWidth: 1.2,
                  targetMarker: { d: "" },
                },
              },
              connector: { name: "smooth" },
            })
          );
        }

        // Second input (dashed) — skip identical
        if (sw !== fw && nodes[l - 1]?.[sw] && nodes[l]?.[w]) {
          graph.addCell(
            new shapes.standard.Link({
              source: { id: nodes[l - 1][sw].id },
              target: { id: nodes[l][w].id },
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
        }
      }
    }

    // 6. CRITICAL: Unfreeze to render
    paper.unfreeze();

    // 7. Fit content after async render
    requestAnimationFrame(() => {
      if (!paperRef.current) return;
      const bbox = graph.getBBox();
      if (!bbox) return;
      paper.setDimensions(bbox.x + bbox.width + 80, bbox.y + bbox.height + 50);
      paper.transformToFitContent({ padding: 30, maxScale: 1.5 });
      setZoomPct(Math.round(paper.scale().sx * 100));
    });

    // 8. Click to inspect gate
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
            <path d="M0 0L9 0C18 0 18 14 9 14L0 14Z" fill="#FEE2E2" stroke="#F0524D" strokeWidth="1.5"/>
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
      </div>

      {/* JointJS canvas — overflow hidden captures wheel for zoom */}
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
