# Circuit Visualization Redesign — Wire-Centric Visual Hierarchy

## Goal

Shift the visual emphasis from gate bodies to wire output ports. The circuit explorer should make E[wire] values the dominant visual element, so that a user can glance at the circuit and immediately read the distribution of wire expectations across layers.

## Design Decisions

### 1. Thin, Low-Opacity Gate Bodies

The gate rectangle shrinks from 48×32px to ~20×32px and drops to ~0.3 opacity. It becomes a subtle structural connector between the two input ports and the output port — present if you look for it, invisible at a glance.

### 2. Larger Output Port Circles

The output port radius grows from 7 to ~11px. This circle is color-coded by `meanToColor(E[wire])` and becomes the **primary visual element** in the graph. Each gate unit is now visually dominated by its colored circle.

### 3. Color-Coded Wires

Wires currently use a uniform gray (`#CBD5E1`). In the redesign, each wire inherits the color of its **source port's E[wire] value**. This makes the color flow through the graph, reinforcing the E[wire] distribution visually.

### 4. Input/Output Wire Endpoints as Filled Dots

Replace the current coral-bordered rectangles (`x0`–`xN`) and slate-bordered rectangles (`y0`–`yN`) with small filled circles (r=5). Labels sit outside the circle (left for inputs, right for outputs). These read as natural wire start/end points without competing with the gate output ports.

### 5. Centralized Tunable Parameters

All dimension and style params live in one place (`gateShapes.js`) so we can rapidly iterate:

```js
export const GATE_W         = 20;    // gate body width
export const GATE_H         = 32;    // gate body height
export const GATE_OPACITY   = 0.3;   // gate body fill opacity
export const WIRE_PORT_R    = 11;    // output port circle radius
export const INPUT_DOT_R    = 5;     // input/output endpoint dot radius
```

### 6. Onboarding Tour (deferred)

A guided tour showing a single gate with two inputs and one output, demonstrating how color changes as information flows through the gate. To be designed and built after the core visual changes are landed and reviewed.

## Visual Summary

```
BEFORE:  [====large gate====] ○  ────gray wire────
         (dominant element)   (small, subtle)

AFTER:   [│]──(████)──════colored wire════──
         ghost   BIG, COLORED           matches port
         gate    output port            color
```
