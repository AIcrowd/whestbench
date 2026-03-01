/**
 * NarrativeCard — Contextual card for each walkthrough step.
 * Shows step-specific text with a colored left border and Back/Next nav.
 */

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
        <strong>Your challenge:</strong> if we feed random ±1 inputs, what is
        the <strong>expected output</strong> E[wire] of each wire? The outputs
        are highlighted on the right (y₀, y₁, …). Each one is a complex
        function of all inputs.
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
        <strong>beat sampling&apos;s accuracy without using more samples.</strong>
      </>
    ),
  },
  5: {
    border: "var(--coral)",
    text: (
      <>
        <strong>Mean Propagation</strong> doesn&apos;t sample at all. It computes
        E[wire] analytically, propagating expected values through each gate&apos;s
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
    <div
      className="narrative-card"
      style={{ borderLeftColor: content.border }}
      key={step}
    >
      <div className="narrative-text">{children || content.text}</div>
      <div className="narrative-nav">
        {step > 1 && (
          <button className="narrative-btn narrative-btn--back" onClick={onBack}>
            ← Back
          </button>
        )}
        {step < 6 && (
          <button
            className="narrative-btn narrative-btn--next"
            onClick={onNext}
          >
            Next →
          </button>
        )}
      </div>
    </div>
  );
}

export { STEP_CONTENT };
