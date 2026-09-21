"use client";

import React from "react";
import { Sun, Moon } from "lucide-react";

export default function Header({ theme, onToggleTheme, adapterInfo }) {
  const hasSize = adapterInfo?.sizeMb !== undefined && adapterInfo?.sizeMb !== null;
  const hasLayers = adapterInfo?.layers !== undefined && adapterInfo?.layers !== null;

  return (
    <header className="h-[72px] border-b border-[var(--line)] bg-[var(--bg)] transition-colors">
      <div className="max-w-[1280px] mx-auto px-4 sm:px-8 h-full flex items-center justify-between">
        {/* Left: Product Name */}
        <div className="flex items-center">
          <span className="text-[28px] leading-[34px] font-[500] tracking-[-0.02em] text-[var(--fg)]">
            Mara
          </span>
        </div>

        {/* Right: Badge Meta & Theme Toggle */}
        <div className="flex items-center space-x-6 text-[13px] leading-[18px] text-[var(--fg-2)] font-normal">
          <div className="flex items-center space-x-6">
            <span
              title={
                adapterInfo?.kind === "browser-wasm"
                  ? "Running client-side in browser via WebAssembly, downloaded directly from Hugging Face Hub."
                  : adapterInfo?.kind === "on-device"
                  ? "Connected to live local Mara-AFM PyTorch engine (692k params, CPU)."
                  : "Answers come from built-in simulation, not a running model."
              }
              className="cursor-help flex items-center gap-1.5"
            >
              {adapterInfo?.kind === "browser-wasm" && (
                <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              )}
              {adapterInfo?.kind === "browser-wasm"
                ? "In-Browser (HF WASM)"
                : adapterInfo?.kind === "on-device"
                ? "Live PyTorch"
                : "Simulated"}
            </span>

            {hasSize && <span>{adapterInfo.sizeMb} MB</span>}
            {hasLayers && <span>{adapterInfo.layers} layers</span>}
          </div>

          <button
            type="button"
            onClick={onToggleTheme}
            aria-label="Switch theme"
            className="w-8 h-8 rounded-[6px] border border-[var(--line)] flex items-center justify-center text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors cursor-pointer"
          >
            {theme === "dark" ? (
              <Sun size={16} strokeWidth={1.5} />
            ) : (
              <Moon size={16} strokeWidth={1.5} />
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
