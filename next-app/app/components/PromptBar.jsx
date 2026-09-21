"use client";

import React, { useRef, useEffect } from "react";
import { CornerDownLeft } from "lucide-react";

export default function PromptBar({
  prompt,
  onChangePrompt,
  onSubmit,
  placeholder,
  isLoading,
}) {
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;
    onSubmit(prompt.trim());
  };

  const isDisabled = !prompt.trim() || isLoading;

  return (
    <form onSubmit={handleSubmit} className="relative w-full">
      <div className="relative flex items-center">
        <input
          ref={inputRef}
          type="text"
          value={prompt}
          onChange={(e) => onChangePrompt(e.target.value)}
          placeholder={placeholder}
          disabled={isLoading}
          className="w-full h-[56px] pl-4 pr-[54px] rounded-[6px] border border-[var(--line-strong)] bg-[var(--bg)] text-[var(--fg)] placeholder-[var(--fg-3)] text-[16px] leading-[24px] focus:outline-none focus:border-[var(--fg)] transition-colors disabled:opacity-50"
        />
        <div className="absolute right-2.5 flex items-center">
          <button
            type="submit"
            disabled={isDisabled}
            aria-label="Submit prompt"
            className={`w-[36px] h-[36px] rounded-[6px] flex items-center justify-center transition-colors cursor-pointer ${
              isDisabled
                ? "bg-[var(--fill-2)] text-[var(--fg-3)] cursor-not-allowed"
                : "bg-[var(--fg)] text-[var(--bg)] hover:opacity-88 active:opacity-76"
            }`}
          >
            <CornerDownLeft size={16} strokeWidth={1.5} />
          </button>
        </div>
      </div>
    </form>
  );
}
