/**
 * StepIndicator — Horizontal 6-step progress bar.
 * Active step = coral filled, completed = coral outline + checkmark, future = gray.
 */
export default function StepIndicator({ currentStep, onSkipTour }) {
  const steps = [
    { num: 1, label: "The MLP" },
    { num: 2, label: "Forward Pass" },
    { num: 3, label: "Neuron Means" },
    { num: 4, label: "Sampling" },
    { num: 5, label: "Mean Prop" },
    { num: 6, label: "Challenge" },
  ];

  return (
    <div className="step-indicator">
      <div className="step-track">
        {steps.map((s, i) => (
          <div key={s.num} className="step-item-wrapper">
            {i > 0 && (
              <div
                className={`step-connector ${
                  currentStep > s.num - 1 ? "step-connector--done" : ""
                }`}
              />
            )}
            <div
              className={`step-circle ${
                currentStep === s.num
                  ? "step-circle--active"
                  : currentStep > s.num
                    ? "step-circle--done"
                    : ""
              }`}
            >
              {currentStep > s.num ? (
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              ) : (
                s.num
              )}
            </div>
            <span
              className={`step-label ${
                currentStep === s.num ? "step-label--active" : ""
              }`}
            >
              {s.label}
            </span>
          </div>
        ))}
      </div>
      {currentStep < 6 && (
        <button className="skip-tour-btn" onClick={() => onSkipTour()}>
          Skip tour →
        </button>
      )}
      {currentStep === 6 && (
        <button
          className="skip-tour-btn"
          onClick={() => onSkipTour("restart")}
        >
          📖 Restart tour
        </button>
      )}
    </div>
  );
}
