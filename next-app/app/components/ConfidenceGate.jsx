"use client";

import React, { useEffect, useState, useRef } from "react";

export default function ConfidenceGate({ confidence = 0, outcome = "refuse", actAt = 0.7 }) {
  const [displayScore, setDisplayScore] = useState(0);
  const [progress, setProgress] = useState(0); // 0 to 1
  const animRef = useRef(null);

  useEffect(() => {
    // Check prefers-reduced-motion
    const prefersReducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const targetScore = Math.max(0, Math.min(1, confidence));

    if (prefersReducedMotion) {
      setDisplayScore(targetScore);
      setProgress(targetScore);
      return;
    }

    // Animate from 0 to targetScore over 420ms using cubic-bezier(0.2, 0, 0, 1)
    const duration = 420;
    const startTime = performance.now();

    const cubicBezier = (t) => {
      // Approximation of cubic-bezier(0.2, 0, 0, 1)
      return t === 1 ? 1 : 1 - Math.pow(1 - t, 3.5);
    };

    const step = (now) => {
      const elapsed = now - startTime;
      const t = Math.min(1, elapsed / duration);
      const easedT = cubicBezier(t);
      const currentVal = easedT * targetScore;

      setDisplayScore(currentVal);
      setProgress(currentVal);

      if (t < 1) {
        animRef.current = requestAnimationFrame(step);
      } else {
        setDisplayScore(targetScore);
        setProgress(targetScore);
      }
    };

    animRef.current = requestAnimationFrame(step);

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [confidence]);

  const outcomeLabels = {
    act: "Run",
    confirm: "Ask first",
    refuse: "Refuse",
  };

  const outcomeLabel = outcomeLabels[outcome] || "Refuse";

  const renderOutcomeGlyph = () => {
    switch (outcome) {
      case "act":
        return (
          <svg width="12" height="12" viewBox="0 0 12 12" className="text-[var(--fg)] shrink-0">
            <circle cx="6" cy="6" r="5" fill="currentColor" />
          </svg>
        );
      case "confirm":
        return (
          <svg width="12" height="12" viewBox="0 0 12 12" className="text-[var(--fg)] shrink-0">
            <circle cx="6" cy="6" r="5" fill="none" stroke="currentColor" strokeWidth="1.5" />
            <path d="M6 1 A5 5 0 0 0 6 11 Z" fill="currentColor" />
          </svg>
        );
      case "refuse":
      default:
        return (
          <svg width="12" height="12" viewBox="0 0 12 12" className="text-[var(--fg)] shrink-0">
            <circle
              cx="6"
              cy="6"
              r="5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeDasharray="2.5 2"
            />
          </svg>
        );
    }
  };

  // Convert actAt and 0.1 to percentages for track positioning
  const refuseEndPct = 10; // 0.1 -> 10%
  const actStartPct = Math.round(actAt * 100);
  const fillPct = Math.round(progress * 100);

  return (
    <div
      role="img"
      aria-label={`Confidence ${confidence.toFixed(2)}. ${outcomeLabel}.`}
      className="w-full space-y-3 select-none"
    >
      {/* Top Row: Score & Outcome Label */}
      <div className="flex items-end justify-between">
        <div className="flex items-baseline space-x-2.5">
          <span className="font-mono text-[32px] leading-[36px] font-[400] text-[var(--fg)] tracking-[-0.02em] tabular-nums">
            {displayScore.toFixed(2)}
          </span>
          <span className="text-[13px] leading-[18px] text-[var(--fg-2)] font-normal">
            Confidence
          </span>
        </div>

        <div className="flex items-center space-x-2 text-[14px] leading-[22px] font-[500] text-[var(--fg)]">
          {renderOutcomeGlyph()}
          <span>{outcomeLabel}</span>
        </div>
      </div>

      {/* The Track with 3 Zones and Animated Fill */}
      <div className="relative pt-2 pb-5">
        {/* SVG Track Container */}
        <div className="relative w-full h-[6px] flex items-center">
          {/* Segment 1: 0 to 0.1 (Dashed 1px) */}
          <div
            className="absolute left-0 h-0 border-t border-dashed border-[var(--line-strong)]"
            style={{ width: `${refuseEndPct}%` }}
          />

          {/* Segment 2: 0.1 to actAt (Solid 1px) */}
          <div
            className="absolute h-0 border-t border-solid border-[var(--line-strong)]"
            style={{
              left: `${refuseEndPct}%`,
              width: `${actStartPct - refuseEndPct}%`,
            }}
          />

          {/* Segment 3: actAt to 1 (Solid 3px) */}
          <div
            className="absolute h-0 border-t-[3px] border-solid border-[var(--line-strong)]"
            style={{
              left: `${actStartPct}%`,
              width: `${100 - actStartPct}%`,
            }}
          />

          {/* Fill Line: 3px solid --fg from 0 to fillPct */}
          <div
            className="absolute left-0 h-0 border-t-[3px] border-[var(--fg)] transition-none"
            style={{ width: `${fillPct}%` }}
          />

          {/* End Marker: 10px filled circle */}
          <div
            className="absolute top-1/2 w-[10px] h-[10px] rounded-full bg-[var(--fg)] -translate-x-1/2 -translate-y-1/2 transition-none pointer-events-none"
            style={{ left: `${fillPct}%` }}
          />
        </div>

        {/* Ticks and Labels beneath track */}
        <div className="relative w-full text-[12px] leading-[16px] font-mono text-[var(--fg-2)] mt-2">
          {/* Tick 0 */}
          <span className="absolute left-0 -translate-x-1/2">0</span>

          {/* Tick 0.1 */}
          <div
            className="absolute flex flex-col items-center -translate-x-1/2"
            style={{ left: `${refuseEndPct}%` }}
          >
            <span className="h-1.5 w-[1px] bg-[var(--line-strong)] -mt-1.5 mb-1" />
            <span>0.1</span>
          </div>

          {/* Tick actAt */}
          <div
            className="absolute flex flex-col items-center -translate-x-1/2"
            style={{ left: `${actStartPct}%` }}
          >
            <span className="h-1.5 w-[1px] bg-[var(--line-strong)] -mt-1.5 mb-1" />
            <span className="whitespace-nowrap">{actAt.toFixed(2)} (act at)</span>
          </div>

          {/* Tick 1 */}
          <span className="absolute right-0 translate-x-1/2">1</span>
        </div>
      </div>
    </div>
  );
}
