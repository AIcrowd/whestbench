# Circuit Explorer — Progressive Reveal Redesign

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Restructure the Circuit Explorer dashboard as a 6-step guided walkthrough that teaches the problem progressively, then unlocks full exploration — while improving visual quality.

**Architecture:** Add a step state machine to `App.jsx` that controls component visibility. Two new components (`StepIndicator`, `NarrativeCard`) provide the walkthrough UI. Existing components get `locked`/`autoRun` props. CSS additions handle transitions, locked states, and visual refinements.

**Tech Stack:** React 19 + Vite, JointJS (existing), Recharts (existing), vanilla CSS

**Design doc:** [`.aicrowd/docs/plans/2026-03-01-circuit-explorer-redesign-design.md`](file:///Users/mohanty/work/AIcrowd/challenges/alignment-research-center/circuit-estimation/circuit-estimation-mvp/.aicrowd/docs/plans/2026-03-01-circuit-explorer-redesign-design.md)

---

## Task 1: StepIndicator Component

**Files:**
- Create: `tools/circuit-explorer/src/components/StepIndicator.jsx`
- Modify: `tools/circuit-explorer/src/App.css` (add step indicator styles)

**Step 1: Create StepIndicator component**

```jsx
// StepIndicator.jsx
// Horizontal 6-step progress bar: numbered circles connected by lines.
// Active step = coral filled, completed = coral outline + checkmark, future = gray.
// Props: currentStep (1-6), onStepClick, onSkip

export default function StepIndicator({ currentStep, onSkipTour }) {
  const steps = [
    { num: 1, label: "The Circuit" },
    { num: 2, label: "The Problem" },
    { num: 3, label: "Brute Force" },
    { num: 4, label: "On a Budget" },
    { num: 5, label: "Mechanistic" },
    { num: 6, label: "Explore" },
  ];

  return (
    <div className="step-indicator">
      <div className="step-track">
        {steps.map((s, i) => (
          <div key={s.num} className="step-item-wrapper">
            {i > 0 && (
              <div className={`step-connector ${currentStep > s.num - 1 ? "step-connector--done" : ""}`} />
            )}
            <div className={`step-circle ${currentStep === s.num ? "step-circle--active" : currentStep > s.num ? "step-circle--done" : ""}`}>
              {currentStep > s.num ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
              ) : s.num}
            </div>
            <span className={`step-label ${currentStep === s.num ? "step-label--active" : ""}`}>{s.label}</span>
          </div>
        ))}
      </div>
      {currentStep < 6 && (
        <button className="skip-tour-btn" onClick={onSkipTour}>
          Skip tour →
        </button>
      )}
      {currentStep === 6 && (
        <button className="skip-tour-btn" onClick={() => onSkipTour("restart")}>
          📖 Restart tour
        </button>
      )}
    </div>
  );
}
```

**Step 2: Add CSS for StepIndicator**

Add to `App.css`:

```css
/* ──── Step Indicator ──── */
.step-indicator {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
  background: var(--white);
  border-bottom: 1px solid var(--gray-200);
}

.step-track {
  display: flex;
  align-items: center;
  gap: 0;
}

.step-item-wrapper {
  display: flex;
  align-items: center;
  gap: 0;
}

.step-connector {
  width: 40px;
  height: 2px;
  background: var(--gray-200);
  margin: 0 4px;
  transition: background 0.3s;
}

.step-connector--done {
  background: var(--coral);
}

.step-circle {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 700;
  font-family: var(--font-mono);
  border: 2px solid var(--gray-200);
  color: var(--gray-400);
  background: var(--white);
  transition: all 0.3s;
  flex-shrink: 0;
}

.step-circle--active {
  background: var(--coral);
  border-color: var(--coral);
  color: var(--white);
  box-shadow: 0 0 0 4px var(--coral-light);
}

.step-circle--done {
  background: var(--white);
  border-color: var(--coral);
  color: var(--coral);
}

.step-label {
  font-size: 10px;
  color: var(--gray-400);
  margin-left: 6px;
  margin-right: 8px;
  white-space: nowrap;
  font-weight: 500;
}

.step-label--active {
  color: var(--coral);
  font-weight: 600;
}

.skip-tour-btn {
  background: none;
  border: 1px solid var(--gray-200);
  border-radius: var(--radius-pill);
  padding: 5px 14px;
  font-family: var(--font-sans);
  font-size: 11px;
  font-weight: 500;
  color: var(--gray-400);
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}

.skip-tour-btn:hover {
  border-color: var(--coral);
  color: var(--coral);
}
```

**Step 3: Verify in browser**

Run: `npm run dev` (should already be running)  
Open: `http://localhost:5174/`  
Expected: No visible change yet (component not imported)

**Step 4: Commit**

```bash
cd tools/circuit-explorer
git add src/components/StepIndicator.jsx src/App.css
git commit -m "feat(explorer): add StepIndicator component with CSS"
```

---

## Task 2: NarrativeCard Component

**Files:**
- Create: `tools/circuit-explorer/src/components/NarrativeCard.jsx`
- Modify: `tools/circuit-explorer/src/App.css` (add narrative card styles)

**Step 1: Create NarrativeCard component**

```jsx
// NarrativeCard.jsx
// Contextual card for each walkthrough step.
// Props: step (1-6), onNext, onBack, children (optional extra content)

const STEP_CONTENT = {
  1: {
    border: "var(--gray-400)",
    text: (
      <>
        This is a random Boolean circuit with <strong>4 wires</strong> and{" "}
        <strong>3 layers</strong> of gates. Each gate takes two inputs (±1) and
        computes: <code>out = c + a·x + b·y + p·x·y</code>.{" "}
        <strong>Click any gate</strong> to see its coefficients.
      </>
    ),
  },
  2: {
    border: "var(--gray-400)",
    text: (
      <>
        <strong>Your challenge:</strong> if we feed random ±1 inputs, what is the{" "}
        <strong>expected output</strong> E[wire] of each wire? The outputs are
        highlighted on the right (y₀, y₁, …). Each one is a complex function of
        all inputs.
      </>
    ),
  },
  3: {
    border: "var(--coral)",
    text: null, // filled dynamically with timing data
  },
  4: {
    border: "var(--coral)",
    text: (
      <>
        With only <strong>1,000 samples</strong> instead of 10,000, the estimate
        is faster but noisier. <strong>Try the slider</strong> — watch how MSE
        drops as you increase the budget. The challenge:{" "}
        <strong>beat sampling's accuracy without using more samples.</strong>
      </>
    ),
  },
  5: {
    border: "var(--coral)",
    text: (
      <>
        <strong>Mean Propagation</strong> doesn't sample at all. It computes
        E[wire] analytically, propagating expected values through each gate's
        algebra. <strong>Instant</strong> · No sampling noise · But is it always
        accurate? This is the starting point.{" "}
        <strong>Can you do better?</strong>
      </>
    ),
  },
  6: {
    border: "#10B981",
    text: (
      <>
        🔓 Full explorer mode. Try increasing <strong>depth</strong> to see how
        estimation gets harder. Product-heavy circuits (p·x·y) create non-linear
        dependencies that are hardest to estimate.
      </>
    ),
  },
};

export default function NarrativeCard({ step, onNext, onBack, children }) {
  const content = STEP_CONTENT[step];
  if (!content) return null;

  return (
    <div className="narrative-card" style={{ borderLeftColor: content.border }}>
      <div className="narrative-text">
        {children || content.text}
      </div>
      <div className="narrative-nav">
        {step > 1 && (
          <button className="narrative-btn narrative-btn--back" onClick={onBack}>
            ← Back
          </button>
        )}
        {step < 6 && (
          <button className="narrative-btn narrative-btn--next" onClick={onNext}>
            Next →
          </button>
        )}
      </div>
    </div>
  );
}

export { STEP_CONTENT };
```

**Step 2: Add CSS for NarrativeCard**

Add to `App.css`:

```css
/* ──── Narrative Card ──── */
.narrative-card {
  background: var(--white);
  border: 1px solid var(--gray-200);
  border-left: 4px solid var(--gray-400);
  border-radius: var(--radius-sm);
  padding: 14px 20px;
  max-width: 720px;
  animation: narrative-enter 0.25s ease-out;
}

@keyframes narrative-enter {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.narrative-text {
  font-size: 13.5px;
  line-height: 1.6;
  color: var(--gray-900);
}

.narrative-text strong {
  font-weight: 600;
}

.narrative-text code {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--gray-50);
  border: 1px solid var(--gray-100);
  padding: 1px 6px;
  border-radius: 4px;
  color: var(--gray-600);
}

.narrative-nav {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 12px;
}

.narrative-btn {
  padding: 6px 16px;
  border-radius: var(--radius-pill);
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
}

.narrative-btn--back {
  background: transparent;
  border: 1px solid var(--gray-200);
  color: var(--gray-600);
}

.narrative-btn--back:hover {
  border-color: var(--gray-400);
  color: var(--gray-900);
}

.narrative-btn--next {
  background: var(--coral);
  color: var(--white);
}

.narrative-btn--next:hover {
  background: var(--coral-hover);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(240, 82, 77, 0.25);
}
```

**Step 3: Commit**

```bash
git add src/components/NarrativeCard.jsx src/App.css
git commit -m "feat(explorer): add NarrativeCard component with CSS"
```

---

## Task 3: Add Step State Machine to App.jsx

**Files:**
- Modify: `tools/circuit-explorer/src/App.jsx`

This is the core integration. We add the step state, wire up the two new components, and control visibility.

**Step 1: Update App.jsx with progressive reveal logic**

Replace the entirety of `App.jsx` with the new version. Key changes:

1. Import `StepIndicator` and `NarrativeCard`
2. Add `step` state (1-6), with localStorage for persistence
3. Default circuit params locked to `{ width: 4, depth: 3, seed: 42 }` for steps 1-5, full range at step 6
4. Auto-run estimators at steps 3, 4, 5
5. Conditionally render panels based on step

The tour params are `{ width: 4, depth: 3, seed: 42 }` (small circuit for learning). At step 6, params unlock to the current `DEFAULT_PARAMS`.

```jsx
// Key state additions:
const [step, setStep] = useState(() => {
  const saved = localStorage.getItem("circuit-explorer-tour-step");
  return saved === "done" ? 6 : 1;
});

const tourParams = { width: 4, depth: 3, seed: 42 };
const isTour = step < 6;
const effectiveParams = isTour ? tourParams : params;
```

Auto-run triggers via `useEffect`:
- Step 3: auto-run ground truth
- Step 4: auto-run sampling (budget=1000)
- Step 5: auto-run mean propagation

Visibility rules:
- Steps 1-2: Circuit graph only
- Step 3: Circuit graph (colored) + timing badge
- Step 4: Circuit graph + MSE comparison
- Step 5: Circuit graph + MSE comparison (with mean prop added)
- Step 6: Everything

**Step 2: Verify in browser**

Open `http://localhost:5174/`  
Expected:
- Step indicator bar visible below header (step 1 active)
- Narrative card with "This is a random Boolean circuit…"  message
- 4×3 circuit (not the default 8×6)
- Sidebar controls dimmed
- "Next →" button at bottom of narrative card
- "Skip tour →" in the step indicator

**Step 3: Commit**

```bash
git add src/App.jsx
git commit -m "feat(explorer): integrate progressive reveal step machine"
```

---

## Task 4: Add Locked State to Controls and EstimatorRunner

**Files:**
- Modify: `tools/circuit-explorer/src/components/Controls.jsx`
- Modify: `tools/circuit-explorer/src/components/EstimatorRunner.jsx`

**Step 1: Add `locked` prop to Controls.jsx**

Add a `locked` boolean prop. When true:
- All sliders get `disabled` attribute
- Seed input gets `disabled`
- Regenerate button gets `disabled`
- Wrap the whole panel in `className="controls-panel controls-panel--locked"` when locked
- Add a small lock hint: `<p className="locked-hint">🔒 Unlocks in Explore mode</p>`

**Step 2: Add `autoRun` / `lockedEstimators` props to EstimatorRunner.jsx**

- `lockedEstimators`: array of estimator keys that should be hidden/disabled (e.g., `["groundTruth", "sampling", "meanprop"]` during tour)
- `autoRunKey`: when set, immediately run the corresponding estimator on mount/change

Add `useEffect` that triggers `runEstimator` when `autoRunKey` changes.

**Step 3: Add locked-state CSS**

```css
.controls-panel--locked {
  opacity: 0.4;
  pointer-events: none;
  position: relative;
}

.locked-hint {
  font-size: 10px;
  color: var(--gray-400);
  text-align: center;
  margin-top: 8px;
  font-style: italic;
}
```

**Step 4: Verify in browser**

- Steps 1-5: Controls should appear dimmed with lock hint
- Step 4: Budget slider in EstimatorRunner should be interactive
- Step 6: All controls fully active

**Step 5: Commit**

```bash
git add src/components/Controls.jsx src/components/EstimatorRunner.jsx src/App.css
git commit -m "feat(explorer): add locked states for tour mode controls"
```

---

## Task 5: Output Wire Pulse Animation (Step 2)

**Files:**
- Modify: `tools/circuit-explorer/src/components/CircuitGraphJoint.jsx`
- Modify: `tools/circuit-explorer/src/App.css`

**Step 1: Add `pulseOutputs` prop to CircuitGraphJoint**

When `pulseOutputs` is true, add a CSS class `output-pulse` to the output node SVG elements (y₀, y₁, …). This triggers a subtle scale animation.

In the JointJS element creation for output nodes, conditionally add an animation attribute:
```js
if (pulseOutputs) {
  outNode.attr("body/class", "output-node-pulse");
}
```

**Step 2: Add pulse CSS**

```css
@keyframes output-pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.08); }
}

/* Applied via JointJS class injection */
.output-node-pulse {
  animation: output-pulse 2s ease-in-out infinite;
  transform-origin: center;
}
```

Note: JointJS renders SVG, so CSS animations on SVG attributes may need `transform-box: fill-box`. Test this and adjust if needed.

**Step 3: Verify in browser**

- Navigate to Step 2: output wire markers (y₀, y₁, y₂, y₃) should gently pulse
- Step 1: no pulse; Step 3+: no pulse

**Step 4: Commit**

```bash
git add src/components/CircuitGraphJoint.jsx src/App.css
git commit -m "feat(explorer): add output wire pulse animation for step 2"
```

---

## Task 6: Step 3 Dynamic Narrative (Timing Data)

**Files:**
- Modify: `tools/circuit-explorer/src/App.jsx` (pass timing to NarrativeCard)

**Step 1: Pass ground truth timing to narrative card at step 3**

When the ground truth estimator completes, store the timing result. Pass it as `children` to NarrativeCard for step 3:

```jsx
{step === 3 && (
  <NarrativeCard step={3} onNext={nextStep} onBack={prevStep}>
    {groundTruthTime ? (
      <>
        ✅ We sampled <strong>10,000 random inputs</strong> and averaged each wire.
        The circuit is now colored by E[wire]. Accurate, but took{" "}
        <strong>{formatTime(groundTruthTime)}</strong>. Now imagine{" "}
        <strong>1,000 wires × 256 layers</strong>…
      </>
    ) : (
      <>Computing ground truth…</>
    )}
  </NarrativeCard>
)}
```

**Step 2: Verify in browser**

- Click to step 3: see the narrative card update with actual timing after estimator runs
- Circuit should be colored by mean values

**Step 3: Commit**

```bash
git add src/App.jsx
git commit -m "feat(explorer): dynamic timing in step 3 narrative"
```

---

## Task 7: Panel Reveal Transitions

**Files:**
- Modify: `tools/circuit-explorer/src/App.css`

**Step 1: Add panel enter animation**

```css
/* Panel reveal animation — used for panels that appear on step transitions */
.panel-reveal {
  animation: panel-enter 0.25s ease-out;
}

@keyframes panel-enter {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

**Step 2: Apply `panel-reveal` class to conditionally rendered panels in App.jsx**

Wrap the panels that appear at new steps with this class:
- Step 4: MSE comparison panel gets `panel-reveal`
- Step 6: GateStats, SignalHeatmap, WireStats get `panel-reveal`

**Step 3: Verify in browser**

- Advance through steps: new panels should slide in smoothly from below
- No flicker or layout jump

**Step 4: Commit**

```bash
git add src/App.css src/App.jsx
git commit -m "feat(explorer): add panel reveal transitions"
```

---

## Task 8: Visual Refinements to Existing Panels

**Files:**
- Modify: `tools/circuit-explorer/src/App.css`
- Modify: `tools/circuit-explorer/src/components/EstimatorComparison.jsx` (minor)

**Step 1: Graph container inset styling**

Update the `.joint-container` to have a subtle inset:
```css
.joint-container {
  background: #F9FAFB;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.04);
}
```

**Step 2: MSE panel coral-tinted header**

Add a class `panel--highlight` to EstimatorComparison and style it:
```css
.panel--highlight h2 {
  background: var(--coral-light);
  margin: -16px -20px 16px;
  padding: 12px 20px;
  border-radius: var(--radius) var(--radius) 0 0;
}
```

**Step 3: Improved heatmap gradient legend**

Replace the 3-dot legend in `SignalHeatmap.jsx` with a smooth gradient bar (or keep as-is if time-constrained — this is a polish item).

**Step 4: Header bottom shadow**

```css
.app-header {
  box-shadow: 0 1px 0 rgba(0,0,0,0.04);
}
```

**Step 5: Verify in browser**

- Circuit graph container should have a subtle inset appearance
- MSE panel header should have a coral-tinted background
- Header should have just a whisper of depth

**Step 6: Commit**

```bash
git add src/App.css src/components/EstimatorComparison.jsx
git commit -m "style(explorer): visual refinements — inset graph, coral MSE header"
```

---

## Task 9: Footer CTA and localStorage Tour Persistence

**Files:**
- Modify: `tools/circuit-explorer/src/App.jsx`

**Step 1: Add CTA to footer in step 6**

```jsx
<footer className="app-footer">
  {step === 6 && (
    <p style={{ marginBottom: 4 }}>
      Ready to compete? →{" "}
      <a href="https://www.aicrowd.com/" target="_blank" rel="noopener noreferrer">
        Join the Challenge on AIcrowd
      </a>
    </p>
  )}
  <p>
    Part of the <a href="https://www.alignment.org/" ...>ARC</a> × AIcrowd
    Mechanistic Estimation Challenge
  </p>
</footer>
```

**Step 2: Persist tour completion**

In the `setStep` handler, when step reaches 6:
```js
localStorage.setItem("circuit-explorer-tour-step", "done");
```

On load, check localStorage:
```js
const saved = localStorage.getItem("circuit-explorer-tour-step");
const initialStep = saved === "done" ? 6 : 1;
```

"Restart tour" button: clears localStorage and sets step back to 1.

**Step 3: Verify in browser**

- Complete the tour to step 6
- Refresh the page → should jump to step 6
- Click "📖 Restart tour" → should go back to step 1
- Refresh again → should start at step 1

**Step 4: Commit**

```bash
git add src/App.jsx
git commit -m "feat(explorer): add footer CTA and localStorage tour persistence"
```

---

## Task 10: Build Verification and Lint

**Step 1: Run lint**

```bash
cd tools/circuit-explorer
npm run lint
```

Expected: No errors.

**Step 2: Run production build**

```bash
npm run build
```

Expected: Build succeeds, output in `dist/`.

**Step 3: Run preview**

```bash
npm run preview
```

Open the preview URL. Walk through all 6 steps. Verify everything works in production build.

**Step 4: Commit all remaining changes**

```bash
git add -A
git commit -m "chore(explorer): progressive reveal redesign complete"
```

---

## Verification Plan

### Browser Walkthrough Test (Manual)

After all tasks are complete, verify the following end-to-end flow in the browser at `http://localhost:5174/`:

| Step | Check | Expected |
|------|-------|----------|
| Load | Fresh visit (clear localStorage) | Step 1, 4×3 circuit, locked controls |
| Step 1 | Click a gate | Tooltip appears with coefficients |
| Step 1→2 | Click "Next" | Narrative changes, output wires pulse |
| Step 2→3 | Click "Next" | Ground truth auto-runs, circuit colours by means, timing shown |
| Step 3→4 | Click "Next" | Budget slider appears, sampling auto-runs, MSE panel appears |
| Step 4 | Drag budget slider | MSE updates live |
| Step 4→5 | Click "Next" | Mean prop auto-runs, MSE panel shows both bars |
| Step 5→6 | Click "Next" | All controls unlock, all panels appear, circuit changes to 8×6 |
| Step 6 | Change width/depth | Circuit regenerates, all panels update |
| Step 6 | Click "📖 Restart tour" | Goes back to step 1, circuit returns to 4×3 |
| Any step | Click "Skip tour →" | Jumps to step 6 |
| Refresh | After reaching step 6 | Auto-loads at step 6 |
| Refresh | After "Restart tour" | Starts at step 1 |

### Build Verification

```bash
cd tools/circuit-explorer
npm run lint    # No errors
npm run build   # Builds without errors
```
