/**
 * InfoTip — small ⓘ icon next to panel headers that shows an
 * explanatory tooltip on hover or click.  Pure CSS, no portals.
 * Accepts either a plain `text` string or JSX `children` for
 * rich formatted content.
 */
import React, { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export default function InfoTip({ text, children, trigger }) {
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState(null);
  const tipRef = useRef(null);
  const btnRef = useRef(null);

  const toggle = useCallback((e) => {
    e.stopPropagation();
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      setCoords({
        top: rect.top + rect.height / 2,
        left: rect.right + 10,
      });
    }
    setOpen((o) => !o);
  }, [open]);

  /* close on outside click, Escape, or scroll */
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
    const onScroll = () => setOpen(false);

    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    window.addEventListener("scroll", onScroll, true); // capture phase handles div scrolls
    
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
      window.removeEventListener("scroll", onScroll, true);
    };
  }, [open]);

  const content = children || text;

  const defaultTrigger = (
    <button
      className="info-tip-btn"
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
  );

  const actualTrigger = trigger || defaultTrigger;

  return (
    <span className="info-tip-wrapper">
      {React.cloneElement(actualTrigger, {
        onClick: (e) => {
          if (actualTrigger.props.onClick) actualTrigger.props.onClick(e);
          toggle(e);
        },
        ref: btnRef
      })}
      {open && coords && createPortal(
        <div 
          ref={tipRef} 
          className="info-tip-popup"
          style={{
            position: "fixed",
            top: coords.top,
            left: coords.left,
            transform: "translateY(-50%)",
            margin: 0
          }}
        >
          <div className="info-tip-arrow" />
          <div className="info-tip-body">{content}</div>
        </div>,
        document.body
      )}
    </span>
  );
}
