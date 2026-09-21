"use client";

import React from "react";
import { Pencil, Trash2 } from "lucide-react";

export default function ToolCard({ tool, isJustCalled, onEdit, onRemove }) {
  // Format parameters list e.g. "room, on, brightness?"
  const paramEntries = Object.entries(tool.parameters || {});
  const formattedParams = paramEntries
    .map(([key, spec]) => (spec.required ? key : `${key}?`))
    .join(", ");

  return (
    <div
      className={`group relative rounded-[10px] p-4 bg-[var(--fill-1)] border transition-colors ${
        isJustCalled
          ? "border-l-2 border-l-[var(--fg)] border-t-[var(--line)] border-r-[var(--line)] border-b-[var(--line)]"
          : "border-[var(--line)] hover:border-[var(--line-strong)]"
      }`}
    >
      {/* Row 1: Tool Name & Actions */}
      <div className="flex items-center justify-between mb-1">
        <span className="font-mono text-[13px] leading-[20px] font-[500] text-[var(--fg)]">
          {tool.name}
        </span>
        <div className="flex items-center space-x-1 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity">
          <button
            type="button"
            onClick={onEdit}
            aria-label={`Edit ${tool.name}`}
            className="w-7 h-7 rounded-[4px] flex items-center justify-center text-[var(--fg-2)] hover:text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors cursor-pointer"
          >
            <Pencil size={14} strokeWidth={1.5} />
          </button>
          <button
            type="button"
            onClick={onRemove}
            aria-label={`Remove ${tool.name}`}
            className="w-7 h-7 rounded-[4px] flex items-center justify-center text-[var(--fg-2)] hover:text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors cursor-pointer"
          >
            <Trash2 size={14} strokeWidth={1.5} />
          </button>
        </div>
      </div>

      {/* Row 2: Description */}
      <div className="text-[14px] leading-[22px] text-[var(--fg-2)] line-clamp-2 mb-2 font-normal">
        {tool.description}
      </div>

      {/* Row 3: Inline parameters */}
      <div className="font-mono text-[12px] leading-[16px] text-[var(--fg-2)] truncate">
        {formattedParams || "none"}
      </div>
    </div>
  );
}
