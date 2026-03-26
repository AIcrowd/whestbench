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

const Eneuron = () => (
  <MathTerm
    tip={
      <>
        <strong>E[neuron]</strong> = the <em>expected value</em> (average activation)
        of a neuron over all possible random Gaussian inputs. Because the input
        space is continuous and high-dimensional, we <em>estimate</em> by
        sampling random inputs and averaging.
      </>
    }
  >
    E[neuron]
  </MathTerm>
);

const ReLU = () => (
  <MathTerm
    tip={
      <>
        <strong>ReLU(x)</strong> = max(0, x). Keeps positive values unchanged,
        clips negatives to zero. The standard non-linearity in modern MLPs.
      </>
    }
  >
    ReLU
  </MathTerm>
);

const NormalCDF = () => (
  <MathTerm
    tip={
      <>
        <strong>Φ (normal CDF)</strong> — the cumulative distribution function
        of the standard normal distribution. Φ(z) gives the probability that a
        standard normal random variable is ≤ z.
      </>
    }
  >
    Φ
  </MathTerm>
);

const NormalPDF = () => (
  <MathTerm
    tip={
      <>
        <strong>φ (normal PDF)</strong> — the probability density function
        of the standard normal distribution: φ(z) = (1/√2π)·exp(−z²/2).
      </>
    }
  >
    φ
  </MathTerm>
);

const Covariance = () => (
  <MathTerm
    tip={
      <>
        <strong>Covariance</strong> = Cov(x, y) = E[xy] − E[x]·E[y]. Measures
        how much two neurons move together. Positive = tend to be large/small
        together. Zero = independent. Tracking covariance improves estimation
        accuracy at the cost of O(width²) state per layer.
      </>
    }
  >
    covariance
  </MathTerm>
);

const STEP_CONTENT = {
  1: {
    border: "var(--gray-400)",
    text: (
      <>
        This is a Multi-Layer Perceptron — a stack of layers where each neuron
        computes a weighted sum of its inputs, then applies{" "}
        <ReLU /> (keeping only positive values). The inputs are random Gaussian
        numbers.
      </>
    ),
  },
  2: {
    border: "var(--gray-400)",
    text: (
      <>
        Watch as random inputs flow through the network. Each layer transforms
        the signal — weights amplify or dampen, <ReLU /> clips negatives to
        zero. Notice how the activation pattern changes layer by layer.
      </>
    ),
  },
  3: {
    border: "var(--coral)",
    text: (
      <>
        These colored cells show the average neuron activation across many
        random inputs. Your challenge: predict these averages without running
        thousands of samples. Can you figure out <Eneuron /> from the weights
        alone?
      </>
    ),
  },
  4: {
    border: "var(--coral)",
    text: (
      <>
        The simplest approach: draw random inputs, run them through, average the
        outputs. More samples = better estimates, but it&apos;s slow. Notice the
        noise — with only a small budget, estimates are rough.
      </>
    ),
  },
  5: {
    border: "var(--coral)",
    text: (
      <>
        Instead of sampling, propagate the expected value analytically:{" "}
        <MathTerm
          tip={
            <span className="math-term-tip-body">
              <span className="math-term-tip-row">
                For a pre-activation z ~ N(μ, σ²):<br />
                <strong>E[ReLU(z)] = μΦ(μ/σ) + σφ(μ/σ)</strong>
              </span>
              <span className="math-term-tip-row">
                where Φ is the normal CDF and φ is the normal PDF.
              </span>
              <span className="math-term-tip-row">
                This is exact for Gaussian z, but neurons at layer 2+ are
                not quite Gaussian — so it&apos;s an approximation.
              </span>
            </span>
          }
        >
          E[ReLU(z)] = μ<NormalCDF />(μ/σ) + σ<NormalPDF />(μ/σ)
        </MathTerm>
        . It&apos;s instant — but assumes neurons are independent. Watch it drift
        at deeper layers where correlations build up.
      </>
    ),
  },
  6: {
    border: "#10B981",
    text: (
      <>
        <Covariance /> propagation tracks correlations and does better — but
        costs O(width²) per layer. The contest: given a compute budget, can you
        beat sampling? You now have all the tools. Explore freely!
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

export { Eneuron, STEP_CONTENT };
