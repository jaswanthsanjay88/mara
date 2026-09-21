"use client";

import React, { useState } from "react";
import { Pencil, Trash2, ChevronRight } from "lucide-react";

export default function ToolCard({
  tool,
  isJustCalled,
  onToggle,
  onEdit,
  onRemove,
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isEnabled = tool.enabled !== false;

  // Format parameters list e.g. "room, on, brightness?"
  const paramEntries = Object.entries(tool.parameters || {});
  const formattedParams = paramEntries
    .map(([key, spec]) => (spec.required ? key : `${key}?`))
    .join(", ");

  return (
    <div
      className={`group relative rounded-[8px] p-2.5 bg-[var(--fill-1)] border transition-all duration-150 ${
        !isEnabled
          ? "opacity-55 border-[var(--line)]"
          : isJustCalled
          ? "border-l-2 border-l-[var(--fg)] border-t-[var(--line)] border-r-[var(--line)] border-b-[var(--line)] shadow-sm"
          : "border-[var(--line)] hover:border-[var(--line-strong)]"
      }`}
    >
      {/* Main compact single line */}
      <div className="flex items-center gap-2.5">
        {/* Active Toggle Switch */}
        <button
          type="button"
          role="switch"
          aria-label={`Toggle ${tool.name}`}
          aria-checked={isEnabled}
          onClick={(e) => {
            e.stopPropagation();
            onToggle?.(tool.name, !isEnabled);
          }}
          className={`w-7 h-4 rounded-full p-0.5 transition-colors cursor-pointer shrink-0 ${
            isEnabled ? "bg-[var(--fg)]" : "bg-[var(--line-strong)]"
          }`}
        >
          <div
            className={`w-3 h-3 rounded-full bg-[var(--bg)] transition-transform duration-150 ${
              isEnabled ? "translate-x-3" : "translate-x-0"
            }`}
          />
        </button>

        {/* Name and Params on One Line */}
        <div
          onClick={() => setIsExpanded((prev) => !prev)}
          className="flex-1 min-w-0 flex items-baseline gap-1.5 cursor-pointer select-none overflow-hidden"
          title={`${tool.name}(${formattedParams}): ${tool.description}`}
        >
          <span
            className={`font-mono text-[13px] leading-[18px] font-[500] shrink-0 ${
              isEnabled ? "text-[var(--fg)]" : "text-[var(--fg-3)] line-through"
            }`}
          >
            {tool.name}
          </span>
          <span className="font-mono text-[11px] leading-[16px] text-[var(--fg-3)] truncate">
            ({formattedParams || "none"})
          </span>
        </div>

        {/* Action buttons (Visible on hover) */}
        <div className="flex items-center space-x-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity shrink-0">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onEdit?.();
            }}
            aria-label={`Edit ${tool.name}`}
            className="w-6 h-6 rounded-[4px] flex items-center justify-center text-[var(--fg-2)] hover:text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors cursor-pointer"
          >
            <Pencil size={12} strokeWidth={1.5} />
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onRemove?.();
            }}
            aria-label={`Remove ${tool.name}`}
            className="w-6 h-6 rounded-[4px] flex items-center justify-center text-[var(--fg-2)] hover:text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors cursor-pointer"
          >
            <Trash2 size={12} strokeWidth={1.5} />
          </button>
          <button
            type="button"
            onClick={() => setIsExpanded((prev) => !prev)}
            aria-label="Expand description"
            className="w-6 h-6 rounded-[4px] flex items-center justify-center text-[var(--fg-3)] hover:text-[var(--fg)] transition-transform"
          >
            <ChevronRight
              size={13}
              className={`transition-transform duration-150 ${isExpanded ? "rotate-90" : ""}`}
            />
          </button>
        </div>
      </div>

      {/* Expandable Description Area */}
      {isExpanded && (
        <div className="mt-2 pt-2 border-t border-[var(--line)] text-[12px] leading-[18px] text-[var(--fg-2)] animate-fadeIn">
          <p>{tool.description}</p>
          {paramEntries.length > 0 && (
            <div className="mt-1.5 space-y-0.5">
              <span className="text-[11px] font-mono text-[var(--fg-3)] uppercase tracking-wider">Parameters:</span>
              <ul className="list-disc list-inside space-y-0.5 font-mono text-[11px] text-[var(--fg-2)] pl-1">
                {paramEntries.map(([key, spec]) => (
                  <li key={key}>
                    <span className="text-[var(--fg)]">{key}</span>
                    <span className="text-[var(--fg-3)]"> ({spec.type}{spec.required ? ", required" : ", optional"})</span>
                    {spec.description && <span className="font-sans text-[var(--fg-2)]"> — {spec.description}</span>}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
