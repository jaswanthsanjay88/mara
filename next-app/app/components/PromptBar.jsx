"use client";

import React, { useRef, useEffect } from "react";
import { CornerDownLeft, Loader2 } from "lucide-react";

export default function PromptBar({
  prompt,
  onChangePrompt,
  onSubmit,
  placeholder,
  isLoading,
  downloadProgress,
}) {
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!prompt.trim() || isLoading) return;
    onSubmit(prompt.trim());
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey || !e.shiftKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const isDownloading =
    downloadProgress &&
    downloadProgress.status === "downloading";

  const isDisabled = !prompt.trim() || isLoading;

  return (
    <form onSubmit={handleSubmit} className="relative w-full">
      <div className="relative flex items-center">
        <input
          ref={inputRef}
          type="text"
          value={prompt}
          onChange={(e) => onChangePrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isDownloading ? "Streaming Mara AFM ONNX weights from Hugging Face Hub..." : placeholder}
          disabled={isLoading}
          className="w-full h-[52px] pl-4 pr-[110px] rounded-[8px] border border-[var(--line-strong)] bg-[var(--bg)] text-[var(--fg)] placeholder-[var(--fg-3)] text-[15px] sm:text-[16px] leading-[24px] focus:outline-none focus:border-[var(--fg)] transition-colors disabled:opacity-60 shadow-sm"
        />

        <div className="absolute right-2 flex items-center gap-2">
          {/* Download progress badge */}
          {isDownloading ? (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-[5px] bg-[var(--fill-2)] text-[11px] font-mono text-[var(--fg-2)]">
              <Loader2 size={12} className="animate-spin text-blue-500" />
              <span>{downloadProgress.percent}%</span>
            </div>
          ) : (
            /* Keyboard shortcut hint */
            <span className="hidden sm:inline-block font-mono text-[11px] text-[var(--fg-3)] select-none">
              ⌘↵
            </span>
          )}

          {/* Submit action */}
          <button
            type="submit"
            disabled={isDisabled}
            aria-label="Submit prompt"
            className={`w-[36px] h-[36px] rounded-[6px] flex items-center justify-center transition-all cursor-pointer ${
              isDisabled
                ? "bg-[var(--fill-2)] text-[var(--fg-3)] cursor-not-allowed"
                : "bg-[var(--fg)] text-[var(--bg)] hover:opacity-90 active:scale-95"
            }`}
          >
            {isLoading ? (
              <Loader2 size={15} className="animate-spin" />
            ) : (
              <CornerDownLeft size={15} strokeWidth={1.75} />
            )}
          </button>
        </div>
      </div>
    </form>
  );
}
