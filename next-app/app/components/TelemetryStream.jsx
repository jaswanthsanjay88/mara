"use client";

import React, { useState } from "react";
import { Copy, Check } from "@phosphor-icons/react";

export default function TelemetryStream({ distribution, executionResult, fullState, modelInfo }) {
  const [activeTab, setActiveTab] = useState("execution");
  const [copied, setCopied] = useState(false);

  const tools = [
    "control_device",
    "gpio_write",
    "gpio_read",
    "pwm_set",
    "read_sensor",
    "schedule_timer",
    "none",
  ];

  const getActiveContent = () => {
    switch (activeTab) {
      case "execution":
        return executionResult
          ? JSON.stringify(executionResult, null, 2)
          : "{\n  \"status\": \"ready\"\n}";
      case "state":
        return fullState ? JSON.stringify(fullState, null, 2) : "{}";
      case "model":
        return modelInfo ? JSON.stringify(modelInfo, null, 2) : "{}";
      default:
        return "";
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(getActiveContent());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-[#0a0b10] border border-[#181a24] rounded-md p-5 space-y-4">
      <div className="flex items-center justify-between text-xs font-mono">
        <span className="text-neutral-200 font-medium">Model Telemetry</span>
        <span className="text-neutral-500">Softmax Distribution & Log</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Softmax Probabilities */}
        <div className="p-3.5 rounded border border-[#141620] bg-[#07080c] space-y-2 font-mono text-xs">
          <div className="text-neutral-400 mb-2">Confidence by Tool</div>
          <div className="space-y-1.5">
            {tools.map((name) => {
              const prob = distribution ? (distribution[name] ?? 0) : 0;
              const percent = Math.round(prob * 100);
              const isWinner = prob >= 0.5;

              return (
                <div key={name} className="flex items-center justify-between text-[11px]">
                  <span className={isWinner ? "text-amber-400 font-medium" : "text-neutral-400"}>
                    {name}
                  </span>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-[#141620] h-1 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          isWinner ? "bg-amber-400" : "bg-neutral-600"
                        }`}
                        style={{ width: `${percent}%` }}
                      />
                    </div>
                    <span className="w-7 text-right text-neutral-400">{percent}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* JSON Terminal */}
        <div className="p-3.5 rounded border border-[#141620] bg-[#07080c] space-y-2 font-mono text-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 border-b border-[#141620]">
              <div className="flex items-center space-x-2">
                {[
                  { id: "execution", label: "Dispatch" },
                  { id: "state", label: "State" },
                  { id: "model", label: "Model" },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`text-[11px] px-2 py-0.5 rounded cursor-pointer transition-colors ${
                      activeTab === tab.id
                        ? "bg-[#141622] text-amber-400 border border-[#222738]"
                        : "text-neutral-400 hover:text-neutral-200"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <button
                onClick={handleCopy}
                className="flex items-center space-x-1 text-[11px] text-neutral-400 hover:text-neutral-200 cursor-pointer"
              >
                {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                <span>{copied ? "Copied" : "Copy"}</span>
              </button>
            </div>

            <div className="bg-[#050608] rounded p-2.5 max-h-48 overflow-y-auto text-[11px] text-neutral-300 font-mono mt-2">
              <pre className="whitespace-pre-wrap">{getActiveContent()}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
