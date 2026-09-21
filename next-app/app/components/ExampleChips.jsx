"use client";

import React from "react";

export default function ExampleChips({ chips, onSelectChip, disabled }) {
  return (
    <div className="flex flex-wrap gap-2.5">
      {chips.map((chipText, idx) => {
        const isNegativeTest =
          chipText.toLowerCase().includes("capital of france") ||
          chipText.toLowerCase().includes("france");

        return (
          <button
            key={idx}
            type="button"
            disabled={disabled}
            onClick={() => onSelectChip(chipText)}
            className="group h-[32px] px-3 rounded-[6px] border border-[var(--line)] bg-transparent text-[var(--fg-2)] text-[13px] leading-[18px] font-normal hover:border-[var(--line-strong)] hover:bg-[var(--fill-2)] hover:text-[var(--fg)] transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed max-w-full flex items-center gap-1.5"
          >
            <span className="truncate">{chipText}</span>
            {isNegativeTest && (
              <span className="shrink-0 px-1.5 py-0.5 rounded text-[10px] font-mono tracking-tight bg-[var(--fill-1)] text-[var(--fg-3)] border border-[var(--line)] group-hover:text-[var(--fg-2)] group-hover:border-[var(--line-strong)] transition-colors">
                refusal test · no tool
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
