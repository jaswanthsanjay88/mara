"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import Header from "./components/Header";
import ToolList from "./components/ToolList";
import PromptBar from "./components/PromptBar";
import ExampleChips from "./components/ExampleChips";
import ResponseRegion from "./components/ResponseRegion";
import DevicePanel from "./components/DevicePanel";
import {
  PRESET_CONFIG,
  INITIAL_SMART_HOME_DEVICE_STATE,
  MockAdapter,
  evaluateOutcome,
} from "./lib/mara";
import { browserMara } from "./lib/browserMara";

export default function MaraPlayground() {
  // Theme state
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    // Sync with system or existing theme
    const isDark =
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    const initialTheme = isDark ? "dark" : "light";
    setTheme(initialTheme);
    document.documentElement.setAttribute("data-theme", initialTheme);
    if (initialTheme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  const handleToggleTheme = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    document.documentElement.setAttribute("data-theme", nextTheme);
    if (nextTheme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  // Tools State
  const [tools, setTools] = useState(PRESET_CONFIG["smart-home"].defaultTools);
  const [prompt, setPrompt] = useState("");

  // Response & Model State
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showRunningLine, setShowRunningLine] = useState(false);
  const [isStale, setIsStale] = useState(false);
  const [hasRunManually, setHasRunManually] = useState(false);
  const [error, setError] = useState(null);

  // In-Browser Model Download from Hugging Face Hub
  const [downloadProgress, setDownloadProgress] = useState({
    status: "idle",
    percent: 0,
    text: "",
  });

  // Settings: Act threshold
  const [actAt, setActAt] = useState(0.7);

  // Device States
  const [smartHomeState, setSmartHomeState] = useState(INITIAL_SMART_HOME_DEVICE_STATE);
  const [changedKeys, setChangedKeys] = useState([]);

  // Hover state between Response Call and Floor Plan
  const [hoveredCall, setHoveredCall] = useState(null);

  // Responsive mobile active tab ("request" | "tools" | "device")
  const [mobileTab, setMobileTab] = useState("request");

  // Adapter & Engine state (Live PyTorch vs In-Browser WASM vs Simulated Mock)
  const [engineStatus, setEngineStatus] = useState({
    kind: "simulated",
    sizeMb: 2.7,
    layers: 2,
  });

  const adapterRef = useRef(new MockAdapter());
  const timerRef = useRef(null);

  // Probe live local PyTorch AFM server on mount; fallback to auto-downloading Hugging Face WASM model
  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 350);

    fetch("/api/health", { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        clearTimeout(timeoutId);
        if (data && data.status === "ok") {
          if (isMounted) {
            setEngineStatus({
              kind: "on-device",
              sizeMb: 2.7,
              layers: 2,
            });
          }
        } else {
          throw new Error("No local server");
        }
      })
      .catch(() => {
        if (!isMounted) return;
        // Automatically load in-browser WebAssembly model from Hugging Face Hub
        setDownloadProgress({
          status: "downloading",
          percent: 5,
          text: "Connecting to Hugging Face Hub (jaswanthsanjay88/mara)...",
        });
        browserMara
          .init((p) => {
            if (isMounted) {
              setDownloadProgress(p);
              if (p.status === "ready") {
                setEngineStatus({
                  kind: "browser-wasm",
                  sizeMb: 3.2,
                  layers: 2,
                });
              }
            }
          })
          .catch((err) => {
            console.warn("Browser engine init fallback to simulated:", err);
            if (isMounted) {
              setDownloadProgress({
                status: "error",
                percent: 0,
                text: "Could not load Hugging Face WASM. Running in simulation mode.",
              });
            }
          });
      });

    return () => {
      isMounted = false;
      clearTimeout(timeoutId);
    };
  }, []);

  // Execute calls against Device panel
  const executeCallsOnDevice = useCallback((calls) => {
    if (!calls || calls.length === 0) return;

    let nextHome = {
      ...smartHomeState,
      rooms: { ...smartHomeState.rooms },
      devices: { ...smartHomeState.devices },
      doors: { ...smartHomeState.doors },
    };
    const changed = [];

    for (const call of calls) {
      const { name, arguments: args = {} } = call;

      if (name === "set_lights") {
        const room = args.room || "Living room";
        const matched = Object.keys(nextHome.rooms).find(
          (r) => r.toLowerCase() === String(room).toLowerCase()
        );
        if (matched) {
          const on = args.on !== undefined ? Boolean(args.on) : true;
          const brightness =
            args.brightness !== undefined ? Number(args.brightness) : on ? 100 : 0;
          nextHome.rooms[matched] = { on, brightness };
          changed.push(matched);
        }
      } else if (name === "set_thermostat") {
        if (args.temperature_c !== undefined) {
          nextHome.thermostat = Number(args.temperature_c);
          changed.push("thermostat");
        }
      } else if (name === "control_device") {
        const dev = args.device || "Fan";
        const matched = Object.keys(nextHome.devices).find((d) =>
          d.toLowerCase().includes(String(dev).toLowerCase())
        );
        if (matched) {
          let action = String(args.action || "on").toLowerCase();
          let val = action === "on" ? "On" : "Off";
          if (matched === "Garage door") {
            val = action === "on" || action === "open" ? "Open" : "Closed";
          }
          nextHome.devices[matched] = val;
          changed.push(matched);
        }
      } else if (name === "lock_door") {
        const door = args.door || "Front door";
        const matched = Object.keys(nextHome.doors).find((d) =>
          d.toLowerCase().includes(String(door).toLowerCase())
        );
        if (matched) {
          const locked = args.locked !== undefined ? Boolean(args.locked) : true;
          nextHome.doors[matched] = locked ? "Locked" : "Unlocked";
          changed.push(matched);
        }
      }
    }

    setSmartHomeState(nextHome);
    setChangedKeys(changed);

    // Clear flash after 600ms
    setTimeout(() => {
      setChangedKeys([]);
    }, 600);
  }, [smartHomeState]);

  // Submit Prompt
  const handleSubmitPrompt = useCallback(
    async (textToSubmit) => {
      if (!textToSubmit || isLoading) return;
      setIsLoading(true);
      setShowRunningLine(false);
      setError(null);
      setHasRunManually(false);
      setIsStale(false);

      // Section 6.3: If adapter hasn't answered within 150ms, show 1px indeterminate line
      timerRef.current = setTimeout(() => {
        setShowRunningLine(true);
      }, 150);

      try {
        let res = null;

        // If live PyTorch engine is running, route through /api/run
        if (engineStatus.kind === "on-device") {
          try {
            const apiRes = await fetch("/api/run", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ query: textToSubmit }),
            });
            if (apiRes.ok) {
              const data = await apiRes.json();
              if (data && data.status) {
                const calls = [];
                if (data.tool && data.tool !== "none") {
                  if (data.is_plan && data.plan?.steps) {
                    data.plan.steps.forEach((s) => {
                      const mapped = s.tool === "control_device"
                        ? (s.arguments?.device === "lock" ? "lock_door" : s.arguments?.device === "thermostat" ? "set_thermostat" : s.arguments?.device === "light" ? "set_lights" : "control_device")
                        : s.tool;
                      calls.push({ name: mapped, arguments: s.arguments || {} });
                    });
                  } else {
                    const mapped = data.tool === "control_device"
                      ? (data.arguments?.device === "lock" ? "lock_door" : data.arguments?.device === "thermostat" ? "set_thermostat" : data.arguments?.device === "light" ? "set_lights" : "control_device")
                      : data.tool;
                    calls.push({ name: mapped, arguments: data.arguments || {} });
                  }
                }
                res = {
                  function_calls: calls,
                  confidence: data.confidence ?? (calls.length > 0 ? 0.92 : 0.05),
                  reasoning: data.is_plan
                    ? `Live Plan (${data.plan?.total_steps} steps): ${data.plan?.goal}`
                    : `Evaluated live via Mara-AFM PyTorch engine (${data.inference_latency_ms ? Math.round(data.inference_latency_ms) : 9}ms CPU)`,
                  latency_ms: Math.round(data.inference_latency_ms || 9),
                  prefill_tok_s: 4300,
                  decode_tok_s: 0,
                  memory_mb: 2.7,
                };
              }
            }
          } catch (apiErr) {
            console.warn("Live PyTorch engine call failed, falling back to mock:", apiErr);
          }
        }

        // In-Browser WebAssembly inference (Mara ONNX downloaded from Hugging Face Hub)
        if (!res && (engineStatus.kind === "browser-wasm" || browserMara.status === "ready")) {
          try {
            res = await browserMara.complete(textToSubmit, tools, actAt);
          } catch (wasmErr) {
            console.warn("Browser WASM inference failed, falling back to mock:", wasmErr);
          }
        }

        // Fallback to offline MockAdapter
        if (!res) {
          res = await adapterRef.current.complete(textToSubmit, tools);
        }

        clearTimeout(timerRef.current);
        setShowRunningLine(false);
        setResponse(res);

        // Section 6.5: Act outcome executes immediately
        const outcome = evaluateOutcome(res, actAt);
        if (outcome === "act" && res.function_calls && res.function_calls.length > 0) {
          executeCallsOnDevice(res.function_calls);
        }
      } catch (err) {
        clearTimeout(timerRef.current);
        setShowRunningLine(false);
        setError(err.message || "Model execution failed");
      } finally {
        setIsLoading(false);
      }
    },
    [tools, isLoading, actAt, executeCallsOnDevice, engineStatus]
  );

  // Tool modifications mark response stale
  const handleUpdateTool = (oldName, updatedTool) => {
    setTools((prev) => prev.map((t) => (t.name === oldName ? updatedTool : t)));
    if (response) setIsStale(true);
  };

  const handleRemoveTool = (toolName) => {
    setTools((prev) => prev.filter((t) => t.name !== toolName));
    if (response) setIsStale(true);
  };

  const handleAddTool = (newTool) => {
    setTools((prev) => [...prev, newTool]);
    if (response) setIsStale(true);
  };

  // Reset Device state
  const handleResetDevice = () => {
    setSmartHomeState(INITIAL_SMART_HOME_DEVICE_STATE);
    setChangedKeys([]);
    setHoveredCall(null);
  };

  // Confirm Outcome Buttons
  const handleRunConfirm = () => {
    if (response && response.function_calls && response.function_calls.length > 0) {
      executeCallsOnDevice(response.function_calls);
      setHasRunManually(true);
    }
  };

  const handleDismissConfirm = () => {
    setResponse(null);
    setHasRunManually(false);
    setHoveredCall(null);
  };

  // Just-called highlight tools
  const justCalledToolNames = response?.function_calls
    ? response.function_calls.map((c) => c.name)
    : [];

  const currentPresetConfig = PRESET_CONFIG["smart-home"];
  const outcome = evaluateOutcome(response, actAt);

  return (
    <div className="min-h-[100dvh] flex flex-col bg-[var(--bg)] text-[var(--fg)] transition-colors">
      {/* 7.1 Header */}
      <Header
        theme={theme}
        onToggleTheme={handleToggleTheme}
        adapterInfo={{
          kind: engineStatus.kind,
          layers: engineStatus.layers,
          sizeMb: engineStatus.sizeMb,
        }}
      />

      {/* Hugging Face Model Stream / WASM Download Progress Banner */}
      {downloadProgress.status !== "idle" &&
        downloadProgress.status !== "ready" && (
          <div className="w-full bg-[var(--fill-1)] border-b border-[var(--line)] py-2.5 px-4 sm:px-8 text-[13px] leading-[18px] text-[var(--fg-2)] flex flex-wrap items-center justify-between gap-3 transition-all animate-fadeIn">
            <div className="flex items-center gap-2.5">
              <span className={`inline-block w-2 h-2 rounded-full ${
                downloadProgress.status === "error" ? "bg-amber-500" : "bg-blue-500 animate-pulse"
              }`} />
              <span className="text-[var(--fg)] font-[500]">{downloadProgress.text}</span>
            </div>
            {downloadProgress.status === "downloading" && (
              <div className="flex items-center gap-3">
                <div className="w-32 sm:w-44 h-1.5 bg-[var(--line)] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[var(--fg)] transition-all duration-300"
                    style={{ width: `${downloadProgress.percent}%` }}
                  />
                </div>
                <span className="font-mono text-[12px] tabular-nums text-[var(--fg)]">
                  {downloadProgress.percent}%
                </span>
              </div>
            )}
          </div>
        )}

      {/* Mobile Tab Bar (under 700px) */}
      <div className="md:hidden border-b border-[var(--line)] bg-[var(--bg)] px-4">
        <div className="grid grid-cols-3 text-[13px] leading-[18px] text-center font-normal">
          {["request", "tools", "device"].map((tab) => (
            <button
              key={tab}
              onClick={() => setMobileTab(tab)}
              className={`py-2.5 capitalize cursor-pointer border-b-2 transition-colors ${
                mobileTab === tab
                  ? "border-[var(--fg)] text-[var(--fg)] font-[500]"
                  : "border-transparent text-[var(--fg-2)]"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Main 3-Region Layout Shell (Section 4: 300px | flex (min 440px) | 360px) */}
      <main className="flex-1 max-w-[1280px] w-full mx-auto px-4 sm:px-8 py-6">
        <div className="grid grid-cols-1 md:grid-cols-[300px_1fr] lg:grid-cols-[300px_1fr_360px] gap-0 border-0 md:divide-x md:divide-[var(--line)] min-h-[calc(100dvh-130px)]">
          {/* Region 1: Tools (300px) */}
          <section
            className={`md:block pr-0 md:pr-6 pb-6 md:pb-0 ${
              mobileTab === "tools" ? "block" : "hidden"
            }`}
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between pb-1">
                <h2 className="text-[15px] leading-[22px] font-[500] text-[var(--fg)]">
                  Tools
                </h2>
                <span className="text-[12px] font-mono text-[var(--fg-3)]">
                  {tools.length} active
                </span>
              </div>
              <ToolList
                tools={tools}
                justCalledToolNames={justCalledToolNames}
                onUpdateTool={handleUpdateTool}
                onRemoveTool={handleRemoveTool}
                onAddTool={handleAddTool}
              />
            </div>
          </section>

          {/* Region 2: Request & Response (flexible min 440px) */}
          <section
            className={`md:block px-0 md:px-6 pb-6 md:pb-0 space-y-6 min-w-0 ${
              mobileTab === "request" ? "block" : "hidden"
            }`}
          >
            {/* Request Block */}
            <div className="space-y-3">
              <PromptBar
                prompt={prompt}
                onChangePrompt={setPrompt}
                onSubmit={handleSubmitPrompt}
                placeholder={currentPresetConfig.placeholder}
                isLoading={isLoading}
              />
              <ExampleChips
                chips={currentPresetConfig.chips}
                onSelectChip={(chipText) => {
                  setPrompt(chipText);
                  handleSubmitPrompt(chipText);
                }}
                disabled={isLoading}
              />
            </div>

            {/* Hairline Divider */}
            <div className="w-full h-[1px] bg-[var(--line)]" />

            {/* Response Block */}
            <ResponseRegion
              response={response}
              isLoading={isLoading}
              showRunningLine={showRunningLine}
              isStale={isStale}
              actAt={actAt}
              onChangeActAt={setActAt}
              onRunConfirm={handleRunConfirm}
              onDismissConfirm={handleDismissConfirm}
              hasRunManually={hasRunManually}
              error={error}
              onRetry={() => handleSubmitPrompt(prompt)}
              isSimulated={engineStatus.kind === "simulated"}
              onHoverCall={(call) => setHoveredCall(call)}
              onLeaveCall={() => setHoveredCall(null)}
            />

            {/* Tablet fallback: Device section underneath Response (700px to 1099px) */}
            <div className="hidden md:block lg:hidden pt-6 border-t border-[var(--line)]">
              <DevicePanel
                smartHomeState={smartHomeState}
                changedKeys={changedKeys}
                onReset={handleResetDevice}
                activeCalls={response?.function_calls || []}
                outcome={outcome}
                hoveredCall={hoveredCall}
              />
            </div>
          </section>

          {/* Region 3: Device (360px on desktop 1100px+) */}
          <aside
            className={`lg:block pl-0 lg:pl-6 pt-6 lg:pt-0 ${
              mobileTab === "device" ? "block" : "hidden md:hidden lg:block"
            }`}
          >
            <DevicePanel
              smartHomeState={smartHomeState}
              changedKeys={changedKeys}
              onReset={handleResetDevice}
              activeCalls={response?.function_calls || []}
              outcome={outcome}
              hoveredCall={hoveredCall}
            />
          </aside>
        </div>
      </main>
    </div>
  );
}
