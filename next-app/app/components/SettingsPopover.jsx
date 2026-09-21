"use client";

import React, { useState, useRef, useEffect } from "react";
import { SlidersHorizontal } from "lucide-react";

export default function SettingsPopover({ actAt, onChangeActAt }) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  const pct = Math.round(((actAt - 0.1) / (0.95 - 0.1)) * 100);

  return (
    <div ref={containerRef} className="relative inline-block">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Settings"
        aria-expanded={isOpen}
        className="w-7 h-7 rounded-[4px] border border-transparent hover:border-[var(--line)] flex items-center justify-center text-[var(--fg-2)] hover:text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors cursor-pointer"
      >
        <SlidersHorizontal size={15} strokeWidth={1.5} />
      </button>

      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-[260px] p-4 rounded-[10px] border border-[var(--line-strong)] bg-[var(--bg)] shadow-none z-50 space-y-3">
          {/* Header & Value */}
          <div className="flex items-center justify-between text-[13px] leading-[18px]">
            <span className="text-[var(--fg)] font-normal">
              Run without asking above
            </span>
            <span className="font-mono text-[13px] leading-[18px] text-[var(--fg)] font-normal tabular-nums">
              {actAt.toFixed(2)}
            </span>
          </div>

          {/* Range Slider */}
          <div className="relative py-1">
            <input
              type="range"
              min={0.1}
              max={0.95}
              step={0.05}
              value={actAt}
              onChange={(e) => onChangeActAt(parseFloat(e.target.value))}
              className="w-full accent-[var(--fg)] cursor-pointer h-1 bg-[var(--line-strong)] rounded-full"
            />
          </div>

          {/* Helper Text */}
          <div className="text-[12px] leading-[16px] text-[var(--fg-2)] font-normal">
            Below this, the demo asks first. Below 0.1, it refuses.
          </div>
        </div>
      )}
    </div>
  );
}
