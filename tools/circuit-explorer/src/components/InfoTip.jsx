/**
 * InfoTip — small ⓘ icon next to panel headers that shows an
 * explanatory tooltip on hover or click.  Pure CSS, no portals.
 * Accepts either a plain `text` string or JSX `children` for
 * rich formatted content.
 */
import { useCallback, useEffect, useRef, useState } from "react";

export default function InfoTip({ text, children }) {
  const [open, setOpen] = useState(false);
  const tipRef = useRef(null);
  const btnRef = useRef(null);

  const toggle = useCallback((e) => {
    e.stopPropagation();
    setOpen((o) => !o);
  }, []);

  /* close on outside click or Escape */
  useEffect(() => {
    if (!open) return;
    const close = (e) => {
      if (
        tipRef.current &&
        !tipRef.current.contains(e.target) &&
        btnRef.current &&
        !btnRef.current.contains(e.target)
      ) {
        setOpen(false);
      }
    };
    const esc = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  const content = children || text;

  return (
    <span className="info-tip-wrapper">
      <button
        ref={btnRef}
        className="info-tip-btn"
        onClick={toggle}
        aria-label="Show explanation"
        title="What does this mean?"
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
          <text
            x="8"
            y="12"
            textAnchor="middle"
            fontSize="10"
            fontWeight="700"
            fill="currentColor"
          >
            i
          </text>
        </svg>
      </button>
      {open && (
        <div ref={tipRef} className="info-tip-popup">
          <div className="info-tip-arrow" />
          <div className="info-tip-body">{content}</div>
        </div>
      )}
    </span>
  );
}
