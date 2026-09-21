"use client";

import React, { useState } from "react";
import { ArrowBendDownLeft } from "@phosphor-icons/react";

export default function QueryConsole({ onExecute, isLoading, lastQueryMeta }) {
  const [query, setQuery] = useState("");

  const presets = [
    "turn on kitchen lights",
    "turn off kitchen lights and lock front door",
    "good night",
    "set thermostat to 19 and close blinds",
    "movie mode",
    "set GPIO 12 high and read sensor",
    "turn off all lights",
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;
    onExecute(query.trim());
  };

  const handleSelectPreset = (prompt) => {
    setQuery(prompt);
    onExecute(prompt);
  };

  return (
    <div className="space-y-3">
      {/* Command Input */}
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative flex items-center">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Instruct model or select a preset..."
            disabled={isLoading}
            className="w-full bg-[#0a0b10] text-neutral-100 placeholder-neutral-500 text-sm font-mono px-4 py-3 rounded-md border border-[#1a1d28] hover:border-[#282d3e] focus:outline-none focus:border-amber-500/80 transition-colors pr-24 disabled:opacity-50"
          />
          <div className="absolute right-2 flex items-center space-x-1.5">
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="flex items-center space-x-1 px-3 py-1.5 rounded text-xs font-mono font-medium text-neutral-300 hover:text-white bg-[#141622] hover:bg-[#1a1d2c] border border-[#22273a] transition-colors disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
            >
              {isLoading ? (
                <span className="text-amber-400">Running...</span>
              ) : (
                <>
                  <span>Run</span>
                  <kbd className="text-[10px] text-neutral-500 ml-0.5">↵</kbd>
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Minimalist Presets List */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs font-mono text-neutral-500 mr-1">Suggestions:</span>
        {presets.map((prompt, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => handleSelectPreset(prompt)}
            disabled={isLoading}
            className="text-xs font-mono px-2.5 py-1 rounded bg-[#0e1017] hover:bg-[#151824] text-neutral-400 hover:text-neutral-200 border border-[#181b26] transition-colors cursor-pointer disabled:opacity-40"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
