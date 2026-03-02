# Circuit Explorer — Progressive Reveal Redesign

## Problem

The current Circuit Explorer is a developer tool that shows **everything at once** — circuit graph, coefficient charts, estimator buttons — with no narrative. A newcomer sees a circuit diagram and has no idea what they're looking at or why they should care. The tool also feels like a prototype: flat panels, no visual hierarchy, basic HTML controls.

**Goals:**
1. Restructure as a **guided walkthrough** that teaches the problem in 60 seconds, then unlocks full exploration
2. Upgrade the **visual quality** to feel like a production-grade, polished research tool
3. Maintain the current **"Functional Minimalism"** aesthetic from the style guide, but add depth and refinement

---

## Part 1: Interaction Design — 6-Step Progressive Reveal

### Step Controller

- Horizontal step indicator bar across the top of the main content area
- Compact pills: `1 · 2 · 3 · 4 · 5 · 6` with the active step highlighted in coral
- **"Skip tour →"** link on the right for returning users
- `localStorage` remembers completion — returning users auto-skip to step 6
- Back/Next buttons at the bottom of each step's narrative card

### Step Flow

| Step | Title | What appears | Auto-action |
|------|-------|-------------|-------------|
| 1 | "Here's a Circuit" | Circuit graph (4×3), locked controls | None |
| 2 | "The Estimation Problem" | Output wire highlight, problem statement | Output wires pulse |
| 3 | "Brute Force: Just Sample It" | Circuit colors by mean, timing badge | Auto-run ground truth (10k) |
| 4 | "Sampling on a Budget" | Budget slider, MSE comparison panel | Auto-run sampling (1k) |
| 5 | "Analyze the Structure" | Mean propagation result in MSE panel | Auto-run mean propagation |
| 6 | "Explore Freely" | All controls, all panels unlocked | None |

---

## Part 2: Visual & Dashboard Improvements

### 2.1 Header Upgrade

**Current:** Plain white bar with text title and small subtitle.

**Proposed:**
- Keep the clean flat header, but add a **subtle bottom gradient shadow** (just a 1px gradient from `rgba(0,0,0,0.04)` to transparent) for visual separation without violating "no shadows" rule
- Add the **ARC × AIcrowd** logo mark to the left
- Move the subtitle into a compact badge next to the title: `⚡ Circuit Explorer` | `Mechanistic Estimation Challenge`
- Add a "Skip tour →" / "📖 Restart tour" button to the header right

### 2.2 Step Indicator Bar (NEW)

- Full-width bar below the header, above the main content
- Visual style: small numbered circles connected by a thin line
- Active step: coral-filled circle with white number, bold label
- Completed steps: coral-outlined circle with a checkmark
- Future steps: gray-outlined circle with gray number
- Transitions: smooth left-to-right slide when advancing
- Height: ~48px

### 2.3 Narrative Cards (NEW)

- Positioned between step indicator and circuit graph
- Clean white card with a **colored left border** (4px):
  - Steps 1-2: `--gray-400` (neutral/informational)
  - Steps 3-5: `--coral` (action/insight)
  - Step 6: `--data-emerald` (success/unlock)
- Max-width: 720px, centered
- Typography: 14px body text (slightly larger than the current 13px for readability)
- Inline code spans for formulas: `out = c + a·x + b·y + p·x·y`
- Bottom row: **← Back** (secondary button) and **Next →** (primary coral button)
- Smooth slide + fade animation (200ms) on step transition

### 2.4 Sidebar Refinements

**Current issues:** Flat, all controls always visible, estimator descriptions take too much space.

**Proposed:**
- **Locked state** (steps 1-5): controls are visible but dimmed (`opacity: 0.4`), with a `🔒 Unlocks in Explore mode` label. Creates visual anticipation.
- **Progressive unlock:** Budget slider becomes active at step 4 (highlighted with a subtle coral glow)
- **Estimator cards:** More compact, with collapsible descriptions (show on hover/focus)
- **Section dividers:** Replace the flat `border-top` with a more refined spacing + subtle label

### 2.5 Circuit Graph Panel

**Current issues:** The graph panel has no visual weight — it's just a bordered box.

**Proposed:**
- Add a **subtle inset** appearance to the graph container: `background: #F9FAFB` with a subtle `inset box-shadow: inset 0 1px 3px rgba(0,0,0,0.04)`
- The formula legend bar (`out = c + a·x + b·y + p·x·y`) moves **inside** the narrative card for steps 1-3, then pins above the graph in step 6
- Zoom indicator moves to a floating pill in the bottom-right corner of the graph
- **Gate tooltip** gets a refined design: subtle top-left triangle pointer toward the gate, tighter spacing

### 2.6 Data Panels (Heatmap, Charts)

**Current issues:** All panels look identical. No visual hierarchy tells you what matters.

**Proposed:**
- **MSE Comparison** (the "punchline" chart): Gets a subtle **coral-tinted header** (`background: var(--coral-light)`) to draw the eye. This is the key insight panel.
- **Wire Means Heatmap**: Add a smooth gradient bar legend replacing the current 3-dot legend
- **Wire Mean Distribution**: Add subtle fill under the mean line for more visual weight
- **Gate Stats**: Move from always-visible to step 6 only — it's a power-user panel, not essential for learning

### 2.7 Empty State → Welcome State

**Current:** A dashed-border box with "Ready to Explore" text.

**Proposed:** This goes away entirely — replaced by the narrative cards. Step 1 **is** the welcome state, with a much warmer and more interactive feel.

### 2.8 Transitions & Micro-animations

All within the "Functional Minimalism" constraints (no glows, no gradients, no glassmorphism):

- **Panel reveal:** New panels slide in from below with `opacity: 0 → 1` and `translateY(12px → 0)` over 200ms
- **Circuit coloring:** When ground truth runs, gate colors fade in progressively layer-by-layer (20ms stagger per layer) instead of all-at-once
- **Step advance:** The narrative card cross-fades while the step indicator smoothly updates
- **Budget slider live update:** MSE bars animate smoothly instead of snapping to new heights
- **Output wire pulse** (step 2): A subtle scale animation (1.0 → 1.1 → 1.0) with a 2s period on the y₀–y₃ markers

### 2.9 Footer

**Current:** Minimal credits line.

**Proposed:**
- Add **call-to-action** in step 6: "Ready to compete? → Join the Challenge on AIcrowd"
- Keep the ARC × AIcrowd credit
- Add a "View on GitHub" link

---

## Part 3: What Changes vs. Current Code

| Component | Change Type | Description |
|---|---|---|
| `App.jsx` | **MODIFY** | Add step state, progressive visibility logic, auto-run triggers |
| `App.css` | **MODIFY** | Step indicator, narrative card, locked states, transition animations, panel refinements |
| `Controls.jsx` | **MODIFY** | Add `locked` prop to dim controls during tour |
| `EstimatorRunner.jsx` | **MODIFY** | Add `autoRun` + `lockedEstimators` props |
| `CircuitGraphJoint.jsx` | **MODIFY** | Add output wire pulse animation for step 2 |
| `StepIndicator.jsx` | **NEW** | Horizontal step controller component |
| `NarrativeCard.jsx` | **NEW** | Step-specific narrative text + Back/Next buttons |
| `GateStats.jsx` | No change | Just conditionally shown (step 6 only) |
| `SignalHeatmap.jsx` | Minor | Improved gradient legend bar |
| `WireStats.jsx` | No change | Just conditionally shown |
| `EstimatorComparison.jsx` | Minor | Coral-tinted header styling |

## What Stays the Same

- All core logic (`circuit.js`, `estimators.js`)
- The JointJS graph rendering
- The gate tooltip/inspector
- The overall color palette and typography
- The existing style guide principles

---

## Rough Mockups

### Step 3: Guided Walkthrough

![Step 3 mockup — guided walkthrough showing the circuit colored by ground truth means, with a narrative card and step indicator visible](/Users/mohanty/.gemini/antigravity/brain/7186d0c3-75ff-4daf-8bae-9e03bef3b882/dashboard_mockup_step3_1772366701110.png)

### Step 6: Full Explore Mode

![Step 6 mockup — full explore mode with all controls unlocked, showing circuit graph, gate analysis charts, heatmap, wire distribution, and estimation error comparison](/Users/mohanty/.gemini/antigravity/brain/7186d0c3-75ff-4daf-8bae-9e03bef3b882/dashboard_mockup_explore_1772366716177.png)
