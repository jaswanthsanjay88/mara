"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sun, Moon, Rss } from "lucide-react";

export default function Header({ theme: externalTheme, onToggleTheme: externalToggle, adapterInfo }) {
  const pathname = usePathname();
  const [internalTheme, setInternalTheme] = useState("dark");

  useEffect(() => {
    const saved = localStorage.getItem("mara-theme") || "dark";
    setInternalTheme(saved);
  }, []);

  const theme = externalTheme || internalTheme;

  const handleToggleTheme = () => {
    if (externalToggle) {
      externalToggle();
    } else {
      const nextTheme = theme === "dark" ? "light" : "dark";
      setInternalTheme(nextTheme);
      localStorage.setItem("mara-theme", nextTheme);
      document.documentElement.setAttribute("data-theme", nextTheme);
      if (nextTheme === "dark") {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
    }
  };

  const hasSize = adapterInfo?.sizeMb !== undefined && adapterInfo?.sizeMb !== null;
  const hasLayers = adapterInfo?.layers !== undefined && adapterInfo?.layers !== null;

  return (
    <header className="relative z-40 h-[72px] border-b border-[var(--line)] bg-[var(--bg)] transition-colors">
      <div className="max-w-[1280px] mx-auto px-4 sm:px-8 h-full flex items-center justify-between">
        {/* Left: Product Brand & Nav Tabs */}
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2.5 group">
            <span className="text-[24px] sm:text-[26px] leading-[32px] font-[500] tracking-[-0.02em] text-[var(--fg)] select-none">
              Mara
            </span>
            <span className="group/badge relative hidden sm:inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono font-medium text-[var(--fg-3)] bg-[var(--fill-1)] border border-[var(--line)] cursor-help select-none">
              AFM 692k
              <span className="pointer-events-none absolute -bottom-8 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-[5px] bg-[var(--fg)] px-2 py-1 text-[11px] font-medium leading-none text-[var(--bg)] shadow-md opacity-0 -translate-y-1 group-hover/badge:opacity-100 group-hover/badge:translate-y-0 transition-all duration-150 ease-out z-50">
                692k parameters · Atomic Function Model
              </span>
            </span>
          </Link>

          {/* Navigation Links */}
          <nav className="flex items-center gap-1">
            <Link
              href="/"
              className={`px-3 py-1.5 rounded-[6px] text-[13px] font-[500] transition-colors ${
                pathname === "/"
                  ? "text-[var(--fg)] bg-[var(--fill-2)] border border-[var(--line)]"
                  : "text-[var(--fg-2)] hover:text-[var(--fg)] hover:bg-[var(--fill-1)]"
              }`}
            >
              Playground
            </Link>
            <Link
              href="/research"
              className={`px-3 py-1.5 rounded-[6px] text-[13px] font-[500] transition-colors ${
                pathname === "/research"
                  ? "text-[var(--fg)] bg-[var(--fill-2)] border border-[var(--line)]"
                  : "text-[var(--fg-2)] hover:text-[var(--fg)] hover:bg-[var(--fill-1)]"
              }`}
            >
              Research
            </Link>
          </nav>
        </div>

        {/* Right: Badge Meta & External Quick Actions */}
        <div className="flex items-center space-x-3 sm:space-x-5 text-[13px] leading-[18px] text-[var(--fg-2)] font-normal">
          {/* Engine Status info (Only shown if adapterInfo is provided) */}
          {adapterInfo && (
            <div className="hidden md:flex items-center space-x-5">
              <span
                title={
                  adapterInfo?.kind === "browser-wasm"
                    ? "Running client-side in browser via WebAssembly, downloaded directly from Hugging Face Hub."
                    : adapterInfo?.kind === "on-device"
                    ? "Connected to live local Mara-AFM PyTorch engine (692k params, CPU)."
                    : "Answers come from built-in simulation, not a running model."
                }
                className="cursor-help flex items-center gap-1.5 hover:text-[var(--fg)] transition-colors"
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
          )}

          {adapterInfo && <div className="h-4 w-[1px] bg-[var(--line)] hidden md:block" />}

          {/* Action Group: RSS, HF, GitHub & Theme */}
          <div className="flex items-center gap-2">
            {/* RSS Research Blog Feed */}
            <a
              href="/feed.xml"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="RSS Research Feed"
              className="group relative w-8 h-8 rounded-[6px] border border-[var(--line)] flex items-center justify-center text-[var(--fg-2)] hover:text-[var(--fg)] hover:border-[var(--line-strong)] hover:bg-[var(--fill-2)] transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] hover:-translate-y-0.5 active:translate-y-0 active:scale-95 motion-reduce:transform-none motion-reduce:transition-none cursor-pointer"
            >
              <Rss
                size={15}
                strokeWidth={1.75}
                className="transition-all duration-250 ease-out group-hover:scale-115 group-hover:text-[#F26522] dark:group-hover:text-[#F26522] motion-reduce:transform-none"
              />
              {/* Tooltip */}
              <span className="pointer-events-none absolute -bottom-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-[5px] bg-[var(--fg)] px-2 py-1 text-[11px] font-medium leading-none text-[var(--bg)] shadow-md opacity-0 -translate-y-1 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-150 ease-out z-50 hidden sm:block">
                RSS Feed
              </span>
            </a>

            {/* Hugging Face Link */}
            <a
              href="https://huggingface.co/jaswanthsanjay88/mara"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="View Mara on Hugging Face"
              className="group relative w-8 h-8 rounded-[6px] border border-[var(--line)] flex items-center justify-center text-[var(--fg-2)] hover:text-[var(--fg)] hover:border-[var(--line-strong)] hover:bg-[var(--fill-2)] transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] hover:-translate-y-0.5 active:translate-y-0 active:scale-95 motion-reduce:transform-none motion-reduce:transition-none cursor-pointer"
            >
              <svg
                viewBox="0 0 24 24"
                width="16"
                height="16"
                fill="currentColor"
                aria-hidden="true"
                className="transition-all duration-250 ease-out group-hover:scale-115 group-hover:-rotate-6 group-hover:text-[#FFD21E] dark:group-hover:text-[#FFD21E] motion-reduce:transform-none"
              >
                <path d="M12.025 1.13c-5.77 0-10.449 4.647-10.449 10.378 0 1.112.178 2.181.503 3.185.064-.222.203-.444.416-.577a.96.96 0 0 1 .524-.15c.293 0 .584.124.84.284.278.173.48.408.71.694.226.282.458.611.684.951v-.014c.017-.324.106-.622.264-.874s.403-.487.762-.543c.3-.047.596.06.787.203s.31.313.4.467c.15.257.212.468.233.542.01.026.653 1.552 1.657 2.54.616.605 1.01 1.223 1.082 1.912.055.537-.096 1.059-.38 1.572.637.121 1.294.187 1.967.187.657 0 1.298-.063 1.921-.178-.287-.517-.44-1.041-.384-1.581.07-.69.465-1.307 1.081-1.913 1.004-.987 1.647-2.513 1.657-2.539.021-.074.083-.285.233-.542.09-.154.208-.323.4-.467a1.08 1.08 0 0 1 .787-.203c.359.056.604.29.762.543s.247.55.265.874v.015c.225-.34.457-.67.683-.952.23-.286.432-.52.71-.694.257-.16.547-.284.84-.285a.97.97 0 0 1 .524.151c.228.143.373.388.43.625l.006.04a10.3 10.3 0 0 0 .534-3.273c0-5.731-4.678-10.378-10.449-10.378M8.327 6.583a1.5 1.5 0 0 1 .713.174 1.487 1.487 0 0 1 .617 2.013c-.183.343-.762-.214-1.102-.094-.38.134-.532.914-.917.71a1.487 1.487 0 0 1 .69-2.803m7.486 0a1.487 1.487 0 0 1 .689 2.803c-.385.204-.536-.576-.916-.71-.34-.12-.92.437-1.103.094a1.487 1.487 0 0 1 .617-2.013 1.5 1.5 0 0 1 .713-.174m-10.68 1.55a.96.96 0 1 1 0 1.921.96.96 0 0 1 0-1.92m13.838 0a.96.96 0 1 1 0 1.92.96.96 0 0 1 0-1.92M8.489 11.458c.588.01 1.965 1.157 3.572 1.164 1.607-.007 2.984-1.155 3.572-1.164.196-.003.305.12.305.454 0 .886-.424 2.328-1.563 3.202-.22-.756-1.396-1.366-1.63-1.32q-.011.001-.02.006l-.044.026-.01.008-.03.024q-.018.017-.035.036l-.032.04a1 1 0 0 0-.058.09l-.014.025q-.049.088-.11.19a1 1 0 0 1-.083.116 1.2 1.2 0 0 1-.173.18q-.035.029-.075.058a1.3 1.3 0 0 1-.251-.243 1 1 0 0 1-.076-.107c-.124-.193-.177-.363-.337-.444-.034-.016-.104-.008-.2.022q-.094.03-.216.087-.06.028-.125.063l-.13.074q-.067.04-.136.086a3 3 0 0 0-.135.096 3 3 0 0 0-.26.219 2 2 0 0 0-.12.121 2 2 0 0 0-.106.128l-.002.002a2 2 0 0 0-.09.132l-.001.001a1.2 1.2 0 0 0-.105.212q-.013.036-.024.073c-1.139-.875-1.563-2.317-1.563-3.203 0-.334.109-.457.305-.454m.836 10.354c.824-1.19.766-2.082-.365-3.194-1.13-1.112-1.789-2.738-1.789-2.738s-.246-.945-.806-.858-.97 1.499.202 2.362c1.173.864-.233 1.45-.685.64-.45-.812-1.683-2.896-2.322-3.295s-1.089-.175-.938.647 2.822 2.813 2.562 3.244-1.176-.506-1.176-.506-2.866-2.567-3.49-1.898.473 1.23 2.037 2.16c1.564.932 1.686 1.178 1.464 1.53s-3.675-2.511-4-1.297c-.323 1.214 3.524 1.567 3.287 2.405-.238.839-2.71-1.587-3.216-.642-.506.946 3.49 2.056 3.522 2.064 1.29.33 4.568 1.028 5.713-.624m5.349 0c-.824-1.19-.766-2.082.365-3.194 1.13-1.112 1.789-2.738 1.789-2.738s.246-.945.806-.858.97 1.499-.202 2.362c-1.173.864.233 1.45.685.64.451-.812 1.683-2.896 2.322-3.295s1.089-.175.938.647-2.822 2.813-2.562 3.244 1.176-.506 1.176-.506 2.866-2.567 3.49-1.898-.473 1.23-2.037 2.16c-1.564.932-1.686 1.178-1.464 1.53s3.675-2.511 4-1.297c.323 1.214-3.524 1.567-3.287 2.405.238.839 2.71-1.587 3.216-.642.506.946-3.49 2.056-3.522 2.064-1.29.33-4.568 1.028-5.713-.624"/>
              </svg>
              {/* Tooltip */}
              <span className="pointer-events-none absolute -bottom-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-[5px] bg-[var(--fg)] px-2 py-1 text-[11px] font-medium leading-none text-[var(--bg)] shadow-md opacity-0 -translate-y-1 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-150 ease-out z-50 hidden sm:block">
                Hugging Face
              </span>
            </a>

            {/* GitHub Link */}
            <a
              href="https://github.com/jaswanthsanjay88/mara"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="View Mara on GitHub"
              className="group relative h-8 px-2.5 rounded-[6px] border border-[var(--line)] flex items-center gap-1.5 text-[var(--fg-2)] hover:text-[var(--fg)] hover:border-[var(--line-strong)] hover:bg-[var(--fill-2)] transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] hover:-translate-y-0.5 active:translate-y-0 active:scale-95 motion-reduce:transform-none cursor-pointer"
            >
              <svg
                viewBox="0 0 24 24"
                width="15"
                height="15"
                fill="currentColor"
                aria-hidden="true"
                className="transition-all duration-250 ease-out group-hover:scale-110 motion-reduce:transform-none"
              >
                <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
              </svg>
              <span className="hidden sm:inline-block font-mono text-[11px] font-medium text-[var(--fg-3)] group-hover:text-[var(--fg)]">
                ★ Star
              </span>
              {/* Tooltip */}
              <span className="pointer-events-none absolute -bottom-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-[5px] bg-[var(--fg)] px-2 py-1 text-[11px] font-medium leading-none text-[var(--bg)] shadow-md opacity-0 -translate-y-1 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-150 ease-out z-50 hidden sm:block">
                GitHub Repository
              </span>
            </a>

            {/* Theme Toggle Button */}
            <button
              type="button"
              onClick={handleToggleTheme}
              aria-label="Switch theme"
              className="group relative w-8 h-8 rounded-[6px] border border-[var(--line)] flex items-center justify-center text-[var(--fg-2)] hover:text-[var(--fg)] hover:border-[var(--line-strong)] hover:bg-[var(--fill-2)] transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] hover:-translate-y-0.5 active:translate-y-0 active:scale-95 motion-reduce:transform-none motion-reduce:transition-none cursor-pointer"
            >
              {theme === "dark" ? (
                <Sun size={16} strokeWidth={1.5} className="transition-transform duration-300 group-hover:rotate-45 group-hover:scale-110 motion-reduce:transform-none" />
              ) : (
                <Moon size={16} strokeWidth={1.5} className="transition-transform duration-300 group-hover:-rotate-12 group-hover:scale-110 motion-reduce:transform-none" />
              )}
              {/* Tooltip */}
              <span className="pointer-events-none absolute -bottom-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-[5px] bg-[var(--fg)] px-2 py-1 text-[11px] font-medium leading-none text-[var(--bg)] shadow-md opacity-0 -translate-y-1 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-150 ease-out z-50 hidden sm:block">
                {theme === "dark" ? "Light theme" : "Dark theme"}
              </span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
