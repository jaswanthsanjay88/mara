"use client";

import React from "react";
import { CircleAlert } from "lucide-react";
import ConfidenceGate from "./ConfidenceGate";
import OutcomeBar from "./OutcomeBar";
import MetricsStrip from "./MetricsStrip";
import RawJsonViewer from "./RawJsonViewer";
import SettingsPopover from "./SettingsPopover";
import { evaluateOutcome } from "../lib/mara";

export default function ResponseRegion({
  response,
  isLoading,
  showRunningLine,
  isStale,
  actAt,
  onChangeActAt,
  onRunConfirm,
  onDismissConfirm,
  hasRunManually,
  error,
  onRetry,
  isSimulated = true,
  onHoverCall,
  onLeaveCall,
}) {
  const outcome = evaluateOutcome(response, actAt);
  const calls = response?.function_calls || [];
  const suppressed = response?.suppressed_calls || [];
  const displayCalls = calls.length > 0 ? calls : suppressed;

  return (
    <div className="relative w-full space-y-6 pt-1">
      {/* Indeterminate 1px running line along top edge */}
      {isLoading && showRunningLine && <div className="running-line" />}

      {/* Section Heading & Settings */}
      <div className="flex items-center justify-between">
        <h2 className="text-[15px] leading-[22px] font-[500] text-[var(--fg)]">
          Response
        </h2>
        <SettingsPopover actAt={actAt} onChangeActAt={onChangeActAt} />
      </div>

      {/* Global Error State */}
      {error && (
        <div className="py-4 space-y-3">
          <div className="flex items-center space-x-2 text-[14px] leading-[22px] text-[var(--fg)]">
            <CircleAlert size={16} strokeWidth={1.5} className="shrink-0" />
            <span>The model couldn't run. {error}.</span>
          </div>
          <button
            type="button"
            onClick={onRetry}
            className="h-8 px-3 rounded-[6px] border border-[var(--line-strong)] text-[13px] leading-[18px] text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors cursor-pointer"
          >
            Try again
          </button>
        </div>
      )}

      {/* Empty State (Before first run) */}
      {!error && !response && (
        <div className="py-8 text-[14px] leading-[22px] text-[var(--fg-2)] font-normal">
          Ask for something above. Mara returns the tool calls it would make, and how sure it is.
        </div>
      )}

      {/* Active Response Body */}
      {!error && response && (
        <div className="space-y-6">
          {/* Call List */}
          <div className="space-y-3">
            {displayCalls.length === 0 ? (
              <div className="text-[14px] leading-[22px] text-[var(--fg-2)] font-normal">
                No call. None of your tools fit this request.
              </div>
            ) : (
              <div className="divide-y divide-[var(--line)]">
                {displayCalls.map((call, idx) => {
                  const args = Object.entries(call.arguments || {});
                  return (
                    <div
                      key={idx}
                      onMouseEnter={() => onHoverCall?.(call, idx + 1)}
                      onMouseLeave={() => onLeaveCall?.()}
                      onFocus={() => onHoverCall?.(call, idx + 1)}
                      onBlur={() => onLeaveCall?.()}
                      tabIndex={0}
                      className={`py-3.5 flex items-start gap-4 transition-colors rounded-[4px] focus:outline-none focus:bg-[var(--fill-1)] ${
                        idx === 0 ? "pt-0" : ""
                      }`}
                    >
                      {/* Fixed 28px Order Number */}
                      <span className="w-[28px] font-mono text-[12px] leading-[16px] text-[var(--fg-2)] shrink-0 pt-0.5 tabular-nums">
                        {idx + 1}
                      </span>

                      {/* Tool Name & Arguments */}
                      <div className="flex-1 space-y-2">
                        <div className="font-mono text-[14px] leading-[20px] font-[500] text-[var(--fg)]">
                          {call.name}
                        </div>

                        {args.length > 0 && (
                          <div className="grid grid-cols-[120px_1fr] gap-y-1 gap-x-3 text-[13px] leading-[20px] font-mono">
                            {args.map(([k, val]) => {
                              const displayVal =
                                typeof val === "boolean"
                                  ? String(val)
                                  : typeof val === "string"
                                  ? val
                                  : JSON.stringify(val);

                              return (
                                <React.Fragment key={k}>
                                  <span className="text-[var(--fg-2)] truncate">
                                    {k}
                                  </span>
                                  <span className="text-[var(--fg)] tabular-nums break-words">
                                    {displayVal}
                                  </span>
                                </React.Fragment>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Reasoning Line */}
            {response.reasoning && (
              <div className="font-mono text-[12px] leading-[16px] text-[var(--fg-2)] pt-1">
                {response.reasoning}
              </div>
            )}

            {/* Stale Marker */}
            {isStale && (
              <div className="text-[13px] leading-[18px] text-[var(--fg-2)] font-normal">
                Tools changed since this run.
              </div>
            )}
          </div>

          {/* Confidence Gate */}
          <ConfidenceGate
            confidence={response.confidence ?? 0}
            outcome={outcome}
            actAt={actAt}
          />

          {/* Outcome Bar */}
          <OutcomeBar
            outcome={outcome}
            callCount={displayCalls.length}
            onRunConfirm={onRunConfirm}
            onDismissConfirm={onDismissConfirm}
            hasRunManually={hasRunManually}
          />

          {/* Metrics Strip */}
          <MetricsStrip response={response} isSimulated={isSimulated} />

          {/* Raw JSON Toggle */}
          <RawJsonViewer response={response} />

          {/* Polite Screen Reader Announcement */}
          <div aria-live="polite" className="sr-only">
            {`${displayCalls.length} calls, confidence ${(response.confidence ?? 0).toFixed(2)}, ${outcome}`}
          </div>
        </div>
      )}
    </div>
  );
}
