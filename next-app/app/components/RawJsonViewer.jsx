"use client";

import React, { useState } from "react";
import { ChevronDown, Copy, Check } from "lucide-react";

export default function RawJsonViewer({ response }) {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!response) return null;

  const jsonStr = JSON.stringify(response, null, 2);

  const handleCopy = () => {
    navigator.clipboard.writeText(jsonStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  // Helper to format JSON monochrome: keys in --fg, strings/numbers in --fg-2, punctuation in --fg-3
  const renderMonochromeJson = (str) => {
    // Escape HTML chars
    const lines = str.split("\n");
    return lines.map((line, idx) => {
      // Matches "key": value
      const keyValMatch = line.match(/^(\s*)(".*?")(\s*:\s*)(.*)$/);
      if (keyValMatch) {
        const [, indent, key, colon, rest] = keyValMatch;
        return (
          <div key={idx} className="whitespace-pre">
            <span>{indent}</span>
            <span className="text-[var(--fg)]">{key}</span>
            <span className="text-[var(--fg-3)]">{colon}</span>
            <span className="text-[var(--fg-2)]">{rest}</span>
          </div>
        );
      }
      return (
        <div key={idx} className="whitespace-pre text-[var(--fg-3)]">
          {line}
        </div>
      );
    });
  };

  return (
    <div className="w-full space-y-2 pt-1">
      {/* Toggle Button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-1.5 text-[13px] leading-[18px] text-[var(--fg-2)] hover:text-[var(--fg)] transition-colors cursor-pointer"
      >
        <span>Raw JSON</span>
        <ChevronDown
          size={14}
          strokeWidth={1.5}
          className={`transition-transform duration-[var(--dur-base)] ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {/* Code Block Container */}
      {isOpen && (
        <div className="relative surface rounded-[10px] p-4 bg-[var(--fill-1)] border border-[var(--line)] max-h-[320px] overflow-auto">
          {/* Copy Button */}
          <button
            type="button"
            onClick={handleCopy}
            aria-label="Copy JSON"
            className="absolute top-3 right-3 w-7 h-7 rounded-[4px] border border-[var(--line)] flex items-center justify-center text-[var(--fg-2)] hover:text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors cursor-pointer bg-[var(--bg)]"
          >
            {copied ? (
              <Check size={14} strokeWidth={1.5} />
            ) : (
              <Copy size={14} strokeWidth={1.5} />
            )}
          </button>

          {/* Code Body */}
          <div className="font-mono text-[13px] leading-[20px] pr-8">
            {renderMonochromeJson(jsonStr)}
          </div>

          <div aria-live="polite" className="sr-only">
            {copied ? "Copied" : ""}
          </div>
        </div>
      )}
    </div>
  );
}
