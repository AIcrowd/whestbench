/**
 * CircuitGraphJoint — JointJS circuit graph with uniform gates,
 * smooth bezier wiring, and interactive flow highlighting.
 *
 * Each gate computes: output = c + a·x + b·y + p·x·y
 *
 * Highlighting: on click, non-connected elements are dimmed.
 * Connected wires get thicker with arrow markers showing data flow.
 * No gate border color changes — avoids conflict with mean fill colors.
 *
 * Tooltip is draggable.
 */
import { dia, shapes } from "@joint/core";
import { useCallback, useEffect, useRef, useState } from "react";
import { classifyGate, GATE_H, GATE_OPACITY, GATE_TYPES, GATE_W, INPUT_DOT_R, meanToColor, WIRE_PORT_R } from "./gateShapes";

/* ------------------------------------------------------------------ */
/*  Layout                                                             */
/* ------------------------------------------------------------------ */
const COL_GAP = 80;
const ROW_GAP = 10;
const PAD_X = 30;
const PAD_Y = 30;

/* Default wire colors */
const WIRE_COLOR = "#CBD5E1";
const WIRE_FLOW = "#475569";       // single muted color for highlighted wires
const GATE_STROKE = "#1E293B";
const GATE_FILL_DEFAULT = "#FFFFFF";

/* Small arrow marker for flow direction */
const FLOW_MARKER = {
  type: "path",
  d: "M 6 -3 L 0 0 L 6 3 z",
  fill: WIRE_FLOW,
};


/* JointJS port group definitions */
const PORT_GROUPS = {
  in: {
    position: { name: "left" },
    attrs: {
      portBody: { r: 2.5, fill: "#94A3B8", stroke: "none", magnet: false },
    },
    markup: [{ tagName: "circle", selector: "portBody" }],
  },
  out: {
    position: { name: "right" },
    attrs: {
      portBody: {
        r: WIRE_PORT_R,
        fill: GATE_FILL_DEFAULT,
        stroke: GATE_STROKE,
        strokeWidth: 1.5,
        magnet: false,
      },
    },
    markup: [{ tagName: "circle", selector: "portBody" }],
  },
};

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
export default function CircuitGraphJoint({ circuit, means, activeLayer, pulseOutputs }) {
  const canvasRef = useRef(null);
  const paperRef = useRef(null);
  const graphRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [zoomPct, setZoomPct] = useState(100);
  const hasMeans = means && means.some((row) => row && row.some((v) => v !== null && v !== undefined));

  /* Dragging state for tooltip */
  const dragRef = useRef({ dragging: false, startX: 0, startY: 0, origX: 0, origY: 0 });

  const onDragStart = useCallback((e) => {
    // Only drag from header area
    if (!e.target.closest(".tooltip-pro-header")) return;
    e.preventDefault();
    dragRef.current = {
      dragging: true,
      startX: e.clientX,
      startY: e.clientY,
      origX: tooltipPos.x,
      origY: tooltipPos.y,
    };
    const onMove = (me) => {
      if (!dragRef.current.dragging) return;
      const dx = me.clientX - dragRef.current.startX;
      const dy = me.clientY - dragRef.current.startY;
      setTooltipPos({
        x: dragRef.current.origX + dx,
        y: dragRef.current.origY + dy,
      });
    };
    const onUp = () => {
      dragRef.current.dragging = false;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [tooltipPos]);

  /* Helper: reset all highlights */
  function resetHighlights(graph) {
    graph.getElements().forEach((el) => {
      const d = el.get("gateData");
      if (!d) return;
      el.attr("body/opacity", 1);
      el.attr("body/stroke", GATE_STROKE);
      el.attr("body/strokeWidth", 1.5);
      el.attr("label/opacity", 1);
    });
    graph.getLinks().forEach((lk) => {
      lk.attr("line/stroke", WIRE_COLOR);
      lk.attr("line/strokeWidth", 0.8);
      lk.attr("line/opacity", 1);
      lk.attr("line/targetMarker", { d: "" });
    });
  }

  /* Helper: highlight flow — no border changes, just opacity + wire arrows */
  function highlightGate(graph, gateData) {
    const { layerIndex: l, wireIndex: w } = gateData;

    // Dim everything
    graph.getElements().forEach((el) => {
      el.attr("body/opacity", 0.2);
      el.attr("label/opacity", 0.15);
    });
    graph.getLinks().forEach((lk) => {
      lk.attr("line/opacity", 0.05);
    });

    // Restore opacity for the clicked gate (no border change)
    const thisGate = graph.getElements().find((el) => {
      const d = el.get("gateData");
      return d && d.layerIndex === l && d.wireIndex === w;
    });
    if (thisGate) {
      thisGate.attr("body/opacity", 1);
      thisGate.attr("label/opacity", 1);
    }

    // Restore opacity for connected gates (no border change)
    const connectedGates = new Set();
    if (l > 0) {
      connectedGates.add(`${l - 1}-${gateData.first}`);
      connectedGates.add(`${l - 1}-${gateData.second}`);
    }
    if (l < circuit.d - 1) {
      const nextLayer = circuit.gates[l + 1];
      for (let nw = 0; nw < circuit.n; nw++) {
        if (nextLayer.first[nw] === w || nextLayer.second[nw] === w) {
          connectedGates.add(`${l + 1}-${nw}`);
        }
      }
    }

    // Restore connected input nodes (x0, x1, ...) when gate is in layer 0
    const connectedInputs = new Set();
    if (l === 0) {
      connectedInputs.add(gateData.first);
      connectedInputs.add(gateData.second);
    }

    graph.getElements().forEach((el) => {
      const d = el.get("gateData");
      if (d) {
        if (connectedGates.has(`${d.layerIndex}-${d.wireIndex}`)) {
          el.attr("body/opacity", 1);
          el.attr("label/opacity", 1);
        }
      }
      // Restore input nodes feeding the clicked gate
      if (el.get("isInput") && connectedInputs.has(el.get("wireIndex"))) {
        el.attr("body/opacity", 1);
        el.attr("label/opacity", 1);
      }
    });

    // Highlight connected wires with flow arrows
    graph.getLinks().forEach((lk) => {
      const src = lk.source();
      const tgt = lk.target();
      if (!src?.id || !tgt?.id) return;

      const srcEl = graph.getCell(src.id);
      const tgtEl = graph.getCell(tgt.id);
      if (!srcEl || !tgtEl) return;

      const srcD = srcEl.get("gateData");
      const tgtD = tgtEl.get("gateData");

      // Handle input node → layer 0 wires
      if (!srcD && srcEl.get("isInput") && tgtD) {
        const srcWire = srcEl.get("wireIndex");
        if (tgtD.layerIndex === l && tgtD.wireIndex === w && l === 0 &&
            (srcWire === gateData.first || srcWire === gateData.second)) {
          lk.attr("line/stroke", WIRE_FLOW);
          lk.attr("line/strokeWidth", 1.8);
          lk.attr("line/opacity", 1);
          lk.attr("line/targetMarker", FLOW_MARKER);
        }
        return;
      }

      if (!srcD || !tgtD) return;

      const isInput =
        tgtD.layerIndex === l &&
        tgtD.wireIndex === w &&
        srcD.layerIndex === l - 1;

      const isOutput =
        srcD.layerIndex === l &&
        srcD.wireIndex === w &&
        tgtD.layerIndex === l + 1;

      if (isInput || isOutput) {
        lk.attr("line/stroke", WIRE_FLOW);
        lk.attr("line/strokeWidth", 1.8);
        lk.attr("line/opacity", 1);
        lk.attr("line/targetMarker", FLOW_MARKER);
      }
    });
  }

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

    // --- Create input wire endpoint dots (column before layer 0) ---
    const inputNodes = [];
    for (let w = 0; w < circuit.n; w++) {
      const x = PAD_X + INPUT_DOT_R;
      const y = PAD_Y + w * (GATE_H + ROW_GAP) + GATE_H / 2;

      const node = new shapes.standard.Circle({
        position: { x: x - INPUT_DOT_R, y: y - INPUT_DOT_R },
        size: { width: INPUT_DOT_R * 2, height: INPUT_DOT_R * 2 },
        attrs: {
          body: {
            fill: GATE_FILL_DEFAULT,
            stroke: GATE_STROKE,
            strokeWidth: 1,
          },
        },
      });
      node.set("isInput", true);
      node.set("wireIndex", w);
      inputNodes.push(node);
      graph.addCell(node);
    }

    // --- Create gate elements (uniform rectangles) ---
    const GATE_X_OFFSET = INPUT_DOT_R * 2 + COL_GAP; // shift gates right to make room for input dots
    const nodes = [];
    for (let l = 0; l < circuit.d; l++) {
      nodes[l] = [];
      for (let w = 0; w < circuit.n; w++) {
        const mean = means?.[l]?.[w] ?? null;
        const wireFill = meanToColor(mean) || GATE_FILL_DEFAULT;
        const gateInfo = classifyGate(circuit.gates[l], w);

        const x = PAD_X + GATE_X_OFFSET + l * (GATE_W + COL_GAP);
        const y = PAD_Y + w * (GATE_H + ROW_GAP);

        const node = new shapes.standard.Rectangle({
          position: { x, y },
          size: { width: GATE_W, height: GATE_H },
          attrs: {
            body: {
              fill: GATE_FILL_DEFAULT,
              stroke: GATE_STROKE,
              strokeWidth: 1.5,
              rx: 3,
              ry: 3,
              opacity: GATE_OPACITY,
            },
            label: {
              text: gateInfo.symbol,
              fontSize: 9,
              fontFamily: "system-ui, sans-serif",
              fill: gateInfo.color || "#475569",
              fontWeight: 700,
              textAnchor: "middle",
              textVerticalAnchor: "middle",
            },
          },
          ports: {
            groups: PORT_GROUPS,
            items: [
              { id: "in1", group: "in" },
              { id: "in2", group: "in" },
              { id: "out", group: "out", attrs: {
                portBody: {
                  fill: wireFill,
                },
              }},
            ],
          },
        });

        node.set("gateData", {
          layerIndex: l,
          wireIndex: w,
          first: circuit.gates[l].first[w],
          second: circuit.gates[l].second[w],
          const: circuit.gates[l].const[w],
          firstCoeff: circuit.gates[l].firstCoeff[w],
          secondCoeff: circuit.gates[l].secondCoeff[w],
          productCoeff: circuit.gates[l].productCoeff[w],
          mean,
          gateType: gateInfo.type,
          gateLabel: gateInfo.label,
          gateSymbol: gateInfo.symbol,
        });

        nodes[l][w] = node;
        graph.addCell(node);
      }
    }

    // --- Create links from input wires to layer 0 ---
    for (let w = 0; w < circuit.n; w++) {
      const g = circuit.gates[0];
      const fw = g.first[w];
      const sw = g.second[w];

      if (inputNodes[fw] && nodes[0]?.[w]) {
        graph.addCell(
          new shapes.standard.Link({
            source: { id: inputNodes[fw].id },
            target: { id: nodes[0][w].id, port: "in1" },
            attrs: {
              line: {
                stroke: WIRE_COLOR,
                strokeWidth: 0.8,
                targetMarker: { d: "" },
              },
            },
            connector: { name: "smooth" },
          })
        );
      }

      if (inputNodes[sw] && nodes[0]?.[w]) {
        graph.addCell(
          new shapes.standard.Link({
            source: { id: inputNodes[sw].id },
            target: { id: nodes[0][w].id, port: "in2" },
            attrs: {
              line: {
                stroke: WIRE_COLOR,
                strokeWidth: 0.8,
                targetMarker: { d: "" },
              },
            },
            connector: { name: "smooth" },
          })
        );
      }
    }

    // --- Create links between layers (color-coded by source E[wire]) ---
    for (let l = 1; l < circuit.d; l++) {
      for (let w = 0; w < circuit.n; w++) {
        const g = circuit.gates[l];
        const fw = g.first[w];
        const sw = g.second[w];

        if (!nodes[l - 1]?.[fw] || !nodes[l]?.[w]) continue;

        const fwColor = meanToColor(means?.[l - 1]?.[fw] ?? null) || WIRE_COLOR;
        const swColor = meanToColor(means?.[l - 1]?.[sw] ?? null) || WIRE_COLOR;

        // First input
        graph.addCell(
          new shapes.standard.Link({
            source: { id: nodes[l - 1][fw].id, port: "out" },
            target: { id: nodes[l][w].id, port: "in1" },
            attrs: {
              line: {
                stroke: fwColor,
                strokeWidth: 0.8,
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
                stroke: swColor,
                strokeWidth: 0.8,
                targetMarker: { d: "" },
              },
            },
            connector: { name: "smooth" },
          })
        );
      }
    }

    // --- Create output wire endpoint dots (column after last layer) ---
    const lastLayer = circuit.d - 1;
    const outputX = PAD_X + GATE_X_OFFSET + (circuit.d) * (GATE_W + COL_GAP) + INPUT_DOT_R;
    for (let w = 0; w < circuit.n; w++) {
      const y = PAD_Y + w * (GATE_H + ROW_GAP) + GATE_H / 2;

      const outMean = means?.[lastLayer]?.[w] ?? null;
      const outDotFill = meanToColor(outMean) || GATE_FILL_DEFAULT;

      const outNode = new shapes.standard.Circle({
        position: { x: outputX - INPUT_DOT_R, y: y - INPUT_DOT_R },
        size: { width: INPUT_DOT_R * 2, height: INPUT_DOT_R * 2 },
        attrs: {
          body: {
            fill: outDotFill,
            stroke: GATE_STROKE,
            strokeWidth: 1,
            class: pulseOutputs ? "output-node-pulse" : "",
          },
        },
      });
      outNode.set("isOutput", true);
      outNode.set("wireIndex", w);
      graph.addCell(outNode);

      // Link from last layer gate to output dot (color-coded)
      if (nodes[lastLayer]?.[w]) {
        const outWireColor = meanToColor(means?.[lastLayer]?.[w] ?? null) || WIRE_COLOR;
        graph.addCell(
          new shapes.standard.Link({
            source: { id: nodes[lastLayer][w].id, port: "out" },
            target: { id: outNode.id },
            attrs: {
              line: {
                stroke: outWireColor,
                strokeWidth: 0.8,
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

    // Fit content centered
    requestAnimationFrame(() => {
      if (!paperRef.current || !canvasRef.current) return;
      const containerW = canvasRef.current.clientWidth;
      const containerH = canvasRef.current.clientHeight || 500;
      paper.setDimensions(containerW, containerH);
      paper.scaleContentToFit({
        padding: 30,
        maxScale: 1,
      });

      // Center the content horizontally within the container
      const scale = paper.scale().sx;
      const area = paper.getContentArea();
      const contentW = area.width * scale;
      const offsetX = (containerW - contentW) / 2 - area.x * scale;
      paper.translate(offsetX, paper.translate().ty);

      setZoomPct(Math.round(scale * 100));
    });

    // Click handler — highlight flow + show draggable tooltip
    paper.on("element:pointerclick", (view, evt) => {
      const d = view.model.get("gateData");
      if (d) {
        const rect = el.getBoundingClientRect();
        const clientX = evt.clientX || evt.originalEvent?.clientX || 0;
        const clientY = evt.clientY || evt.originalEvent?.clientY || 0;
        setTooltipPos({
          x: clientX - rect.left + 20,
          y: clientY - rect.top - 40,
        });
        setTooltip(d);
        highlightGate(graph, d);
      }
    });
    paper.on("blank:pointerclick", () => {
      setTooltip(null);
      resetHighlights(graph);
    });

    return () => {
      paper.remove();
      paperRef.current = null;
      graphRef.current = null;
    };
  }, [circuit, means, pulseOutputs]);

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

  /* ---- pan (drag to pan) ---- */
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    let panning = false;
    let startX = 0, startY = 0, origTx = 0, origTy = 0;

    const onDown = (e) => {
      // Only pan on blank area (not on gates/links)
      // Check if target is the SVG background or paper element
      if (e.target.closest(".joint-element") || e.target.closest(".joint-link")) return;
      panning = true;
      startX = e.clientX;
      startY = e.clientY;
      const paper = paperRef.current;
      if (paper) {
        const t = paper.translate();
        origTx = t.tx;
        origTy = t.ty;
      }
      el.style.cursor = "grabbing";
    };
    const onMove = (e) => {
      if (!panning) return;
      const paper = paperRef.current;
      if (!paper) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      paper.translate(origTx + dx, origTy + dy);
    };
    const onUp = () => {
      panning = false;
      el.style.cursor = "";
    };
    el.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      el.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  /* ---- Escape key to dismiss tooltip ---- */
  useEffect(() => {
    if (!tooltip) return;
    const onKeyDown = (e) => {
      if (e.key === "Escape") {
        setTooltip(null);
        if (graphRef.current) resetHighlights(graphRef.current);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [tooltip]);

  /* ---- layer dimming (external control) ---- */
  useEffect(() => {
    const g = graphRef.current;
    if (!g || tooltip) return;
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
  }, [activeLayer, tooltip]);

  return (
    <div className="panel circuit-graph-joint" style={{ position: "relative" }}>
      <h2>
        Circuit Structure
        <span className="mode-badge">
          {circuit.n} wires × {circuit.d} layers = {circuit.n * circuit.d} gates
        </span>
      </h2>

      {/* Compact legend bar */}
      <div className="gate-legend" style={{ fontSize: 11, color: "#64748B", alignItems: "center" }}>
        <span>
          Each gate computes: <strong>out = c + a·x + b·y + p·x·y</strong>
        </span>
        {hasMeans && (
          <span style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 16 }}>
            <span style={{ fontSize: 9 }}>−1</span>
            <span style={{
              width: 60, height: 10, borderRadius: 3,
              background: "linear-gradient(to right, #334155, #FFFFFF, #F0524D)",
              border: "1px solid #E5E7EB",
            }} />
            <span style={{ fontSize: 9 }}>+1</span>
            <span style={{ fontSize: 9, color: "#94A3B8" }}>E[wire]</span>
          </span>
        )}
      </div>
      {/* Gate type legend */}
      <div className="gate-legend" style={{ fontSize: 10, color: "#64748B", flexWrap: "wrap", gap: "3px 10px", paddingTop: 0 }}>
        {Object.entries(GATE_TYPES).map(([key, { symbol, color }]) => (
          <span key={key} style={{ display: "inline-flex", alignItems: "center", gap: 2 }}>
            <span style={{ color, fontWeight: 700, fontSize: 11 }}>{symbol}</span>
            <span>{key}</span>
          </span>
        ))}
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

      {/* Draggable tooltip */}
      {tooltip && (
        <div
          className="gate-tooltip-pro"
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
          }}
          onMouseDown={onDragStart}
        >
          <div className="canvas-tip-header" style={{ cursor: "grab", justifyContent: "space-between" }}>
            <span>
              Layer <span className="layer-num">{tooltip.layerIndex}</span>
              {" · "}
              Wire <span className="layer-num">{tooltip.wireIndex}</span>
            </span>
            <button
              className="tooltip-pro-close"
              onClick={() => {
                setTooltip(null);
                if (graphRef.current) resetHighlights(graphRef.current);
              }}
            >
              ×
            </button>
          </div>

          {tooltip.gateLabel && (
            <div className="canvas-tip-rows" style={{ paddingBottom: 2 }}>
              <div className="canvas-tip-row">
                <span className="canvas-tip-label">Gate type</span>
                <span className="canvas-tip-value" style={{ fontWeight: 600, color: GATE_TYPES[tooltip.gateType]?.color || "#475569" }}>
                  {tooltip.gateLabel}
                </span>
              </div>
            </div>
          )}

          <div className="tooltip-pro-formula">
            out = c + a·x + b·y + p·x·y
          </div>

          <div className="canvas-tip-rows">
            <div className="canvas-tip-row">
              <span className="canvas-tip-label">
                <span className="canvas-tip-swatch" style={{ background: "#8B95A2" }} />
                c — constant bias
              </span>
              <span className="canvas-tip-value">{tooltip.const.toFixed(4)}</span>
            </div>
            <div className="canvas-tip-row">
              <span className="canvas-tip-label">
                <span className="canvas-tip-swatch" style={{ background: "#5B7BA8" }} />
                a — first input (x)
              </span>
              <span className="canvas-tip-value">{tooltip.firstCoeff.toFixed(4)}</span>
            </div>
            <div className="canvas-tip-row">
              <span className="canvas-tip-label">
                <span className="canvas-tip-swatch" style={{ background: "#1E293B" }} />
                b — second input (y)
              </span>
              <span className="canvas-tip-value">{tooltip.secondCoeff.toFixed(4)}</span>
            </div>
            <div className="canvas-tip-row">
              <span className="canvas-tip-label">
                <span className="canvas-tip-swatch" style={{ background: "#F0524D" }} />
                p — interaction (x·y)
              </span>
              <span className="canvas-tip-value">{tooltip.productCoeff.toFixed(4)}</span>
            </div>
          </div>

          <div className="canvas-tip-divider" />

          <div className="canvas-tip-rows">
            <div className="canvas-tip-row">
              <span className="canvas-tip-label">← x</span>
              <span className="canvas-tip-value" style={{ fontWeight: 400, fontSize: 10 }}>
                layer {tooltip.layerIndex > 0 ? tooltip.layerIndex - 1 : 'input'}, wire {tooltip.first}
              </span>
            </div>
            <div className="canvas-tip-row">
              <span className="canvas-tip-label">← y</span>
              <span className="canvas-tip-value" style={{ fontWeight: 400, fontSize: 10 }}>
                layer {tooltip.layerIndex > 0 ? tooltip.layerIndex - 1 : 'input'}, wire {tooltip.second}
              </span>
            </div>
          </div>

          {tooltip.mean !== null && (
            <>
              <div className="canvas-tip-divider" />
              <div className="canvas-tip-rows">
                <div className="canvas-tip-row">
                  <span className="canvas-tip-label">E[wire]</span>
                  <span className="canvas-tip-value" style={{ color: "#F0524D" }}>{tooltip.mean.toFixed(4)}</span>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      <div className="zoom-indicator">
        {zoomPct}%
        <span style={{ margin: "0 6px", opacity: 0.3 }}>·</span>
        <span style={{ fontSize: 9, color: "#94A3B8" }}>
          Scroll = zoom · Drag = pan · Click gate = inspect
        </span>
      </div>
    </div>
  );
}
