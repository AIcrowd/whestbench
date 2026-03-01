/**
 * NarrativeCard — Contextual card for each walkthrough step.
 * Shows step-specific text with a colored left border and Back/Next nav.
 */
import { useCallback, useRef, useState } from "react";

/**
 * MathTerm — inline highlighted term with a hover tooltip.
 * Uses position: fixed so the tooltip always appears above all elements.
 */
function MathTerm({ children, tip }) {
  const ref = useRef(null);
  const [pos, setPos] = useState(null);

  const onEnter = useCallback(() => {
    if (!ref.current) return;
    const r = ref.current.getBoundingClientRect();
    setPos({ left: r.left + r.width / 2, top: r.bottom + 8 });
  }, []);

  const onLeave = useCallback(() => setPos(null), []);

  return (
    <span className="math-term" ref={ref} onMouseEnter={onEnter} onMouseLeave={onLeave}>
      {children}
      {pos && (
        <span
          className="math-term-tip math-term-tip--visible"
          style={{ left: pos.left, top: pos.top }}
        >
          {tip}
        </span>
      )}
    </span>
  );
}

const Ewire = () => (
  <MathTerm
    tip={
      <>
        <strong>E[wire]</strong> = the <em>expected value</em> (average output)
        of a wire over all possible ±1 inputs. With <em>n</em> input wires
        there are 2<sup>n</sup> combinations — for 4 wires that&apos;s just 16,
        but for 1,000 wires it&apos;s 2<sup>1000</sup>, far too many to
        enumerate. So we <em>estimate</em> by sampling random inputs and
        averaging.
      </>
    }
  >
    E[wire]
  </MathTerm>
);

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
        the <strong>expected output</strong> <Ewire /> of each wire? The outputs
        are highlighted on the right (y₀, y₁, …). Each one is a complex
        function of all inputs — and the circuit makes it hard to reason about
        analytically.
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
        <MathTerm
          tip={
            <>
              For each gate <code>out = c + a·x + b·y + p·x·y</code>, mean
              propagation computes <strong>E[out] = c + a·E[x] + b·E[y] +
              p·E[x]·E[y]</strong>. It assumes inputs are{" "}
              <em>independent</em> (E[x·y] = E[x]·E[y]) and propagates layer
              by layer. This is exact when wires are truly independent, but
              shared upstream gates create correlations that this method ignores.
            </>
          }
        >
          Mean Propagation
        </MathTerm>{" "}
        doesn&apos;t sample at all. For each gate, it plugs{" "}
        <Ewire /> of the two inputs into the gate formula to get{" "}
        <Ewire /> of the output — layer by layer, front to back.{" "}
        <strong>Instant</strong> · No sampling noise · But it assumes wire
        values are independent, which isn&apos;t always true.{" "}
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

export { Ewire, STEP_CONTENT };

