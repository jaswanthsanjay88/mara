"use client";

import React from "react";

export default function ExampleChips({ chips, onSelectChip, disabled }) {
  return (
    <div className="flex flex-wrap gap-3">
      {chips.map((chipText, idx) => (
        <button
          key={idx}
          type="button"
          disabled={disabled}
          onClick={() => onSelectChip(chipText)}
          className="h-[32px] px-3 rounded-[6px] border border-[var(--line)] bg-transparent text-[var(--fg-2)] text-[13px] leading-[18px] font-normal hover:bg-[var(--fill-2)] hover:text-[var(--fg)] transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed truncate max-w-full"
        >
          {chipText}
        </button>
      ))}
    </div>
  );
}
