"use client";

import React from "react";

export default function MetricsStrip({ response, isSimulated = true }) {
  if (!response) return null;

  const latency = response.latency_ms ?? 66;
  const prefill = response.prefill_tps ? response.prefill_tps.toLocaleString() : "4,300";
  const decode = response.decode_tps ? response.decode_tps.toLocaleString() : "850";
  const mem = response.peak_ram_mb ? response.peak_ram_mb.toFixed(1) : "28.5";

  return (
    <div className="space-y-1.5 pt-1">
      <div className="flex flex-wrap items-center gap-6">
        <div>
          <div className="text-[12px] leading-[16px] text-[var(--fg-2)] font-normal font-sans">
            Latency
          </div>
          <div className="font-mono text-[12px] leading-[16px] text-[var(--fg)] tabular-nums font-normal">
            {latency} ms
          </div>
        </div>

        <div>
          <div className="text-[12px] leading-[16px] text-[var(--fg-2)] font-normal font-sans">
            Prefill
          </div>
          <div className="font-mono text-[12px] leading-[16px] text-[var(--fg)] tabular-nums font-normal">
            {prefill} tok/s
          </div>
        </div>

        <div>
          <div className="text-[12px] leading-[16px] text-[var(--fg-2)] font-normal font-sans">
            Decode
          </div>
          <div className="font-mono text-[12px] leading-[16px] text-[var(--fg)] tabular-nums font-normal">
            {decode} tok/s
          </div>
        </div>

        <div>
          <div className="text-[12px] leading-[16px] text-[var(--fg-2)] font-normal font-sans">
            Memory
          </div>
          <div className="font-mono text-[12px] leading-[16px] text-[var(--fg)] tabular-nums font-normal">
            {mem} MB
          </div>
        </div>
      </div>

      {isSimulated ? (
        <div className="text-[12px] leading-[16px] text-[var(--fg-2)] font-normal font-sans">
          Timings are simulated.
        </div>
      ) : response?.is_browser_wasm ? (
        <div className="text-[12px] leading-[16px] text-[var(--fg-2)] font-normal font-sans flex items-center gap-1.5 pt-0.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <span>Executed client-side via WebAssembly · Model streamed from Hugging Face</span>
        </div>
      ) : (
        <div className="text-[12px] leading-[16px] text-[var(--fg-2)] font-normal font-sans flex items-center gap-1.5 pt-0.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500" />
          <span>Executed on local Mara-AFM PyTorch CPU backend</span>
        </div>
      )}
    </div>
  );
}
