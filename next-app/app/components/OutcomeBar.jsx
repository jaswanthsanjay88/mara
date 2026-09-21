"use client";

import React from "react";

export default function OutcomeBar({
  outcome,
  callCount,
  onRunConfirm,
  onDismissConfirm,
  hasRunManually = false,
}) {
  const countText = callCount === 1 ? "1 call" : `${callCount} calls`;

  return (
    <div className="w-full border-t border-[var(--line)] py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-[14px] leading-[22px] transition-colors">
      <div>
        {outcome === "act" || hasRunManually ? (
          <span className="text-[var(--fg)]">Ran {countText}.</span>
        ) : outcome === "confirm" ? (
          <span className="text-[var(--fg)]">
            Not sure enough to run without asking.
          </span>
        ) : (
          <span className="text-[var(--fg)]">I can't do that here.</span>
        )}
      </div>

      {outcome === "confirm" && !hasRunManually && (
        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={onDismissConfirm}
            className="h-[36px] px-3.5 rounded-[6px] border border-[var(--line-strong)] bg-transparent text-[var(--fg)] text-[14px] leading-[22px] font-[500] hover:bg-[var(--fill-2)] transition-colors cursor-pointer"
          >
            Dismiss
          </button>
          <button
            type="button"
            onClick={onRunConfirm}
            className="h-[36px] px-3.5 rounded-[6px] bg-[var(--fg)] text-[var(--bg)] text-[14px] leading-[22px] font-[500] hover:opacity-88 active:opacity-76 transition-opacity cursor-pointer"
          >
            Run
          </button>
        </div>
      )}
    </div>
  );
}
