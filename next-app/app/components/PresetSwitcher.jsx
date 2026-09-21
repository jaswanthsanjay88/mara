"use client";

import React from "react";

export default function PresetSwitcher({ activePreset, onSelectPreset }) {
  const options = [
    { id: "smart-home", label: "Smart home" },
    { id: "computer", label: "Computer" },
  ];

  const handleKeyDown = (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault();
      const nextIdx = (options.findIndex((o) => o.id === activePreset) + 1) % options.length;
      onSelectPreset(options[nextIdx].id);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault();
      const prevIdx =
        (options.findIndex((o) => o.id === activePreset) - 1 + options.length) % options.length;
      onSelectPreset(options[prevIdx].id);
    }
  };

  return (
    <div
      role="radiogroup"
      aria-label="Preset"
      onKeyDown={handleKeyDown}
      className="h-[32px] w-full border border-[var(--line-strong)] rounded-[6px] p-[2px] grid grid-cols-2 gap-1 bg-transparent"
    >
      {options.map((opt) => {
        const isSelected = activePreset === opt.id;
        return (
          <button
            key={opt.id}
            type="button"
            role="radio"
            aria-checked={isSelected}
            tabIndex={isSelected ? 0 : -1}
            onClick={() => onSelectPreset(opt.id)}
            className={`h-full flex items-center justify-center text-[13px] leading-[18px] font-[500] rounded-[5px] transition-colors cursor-pointer ${
              isSelected
                ? "bg-[var(--fg)] text-[var(--bg)]"
                : "bg-transparent text-[var(--fg-2)] hover:bg-[var(--fill-2)] hover:text-[var(--fg)]"
            }`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
