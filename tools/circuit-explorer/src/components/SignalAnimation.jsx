/**
 * SignalAnimationButton — Trigger button for animated signal flow.
 * The hook (useSignalAnimation) is in hooks/useSignalAnimation.js
 * to allow React Fast Refresh to work properly.
 */

export default function SignalAnimationButton({ onAnimate, isAnimating }) {
  return (
    <button
      className="animate-btn"
      onClick={onAnimate}
      disabled={isAnimating}
      title="Animate a single random input flowing through the circuit"
    >
      {isAnimating ? (
        <span className="animate-btn-pulse">◉ Flowing…</span>
      ) : (
        <>
          <svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none">
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          Animate Flow
        </>
      )}
    </button>
  );
}
