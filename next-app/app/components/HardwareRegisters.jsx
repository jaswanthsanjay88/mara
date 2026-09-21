"use client";

import React from "react";

export default function HardwareRegisters({ gpioState, pwmState, onTriggerAction, isLoading }) {
  const gpio = gpioState || { "12": 1, "14": 0, "27": 0 };
  const pwm = (pwmState && pwmState["18"]) || { duty: 0, freq: 1000 };
  const duty = pwm.duty ?? 0;

  return (
    <div className="bg-[#0a0b10] border border-[#181a24] rounded-md p-5 space-y-4">
      <div className="flex items-center justify-between text-xs font-mono">
        <span className="text-neutral-200 font-medium">Hardware Bus</span>
        <span className="text-neutral-500">MCU 240MHz</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* GPIO Pins */}
        <div className="p-3.5 rounded border border-[#141620] bg-[#07080c] space-y-2.5 font-mono text-xs">
          <div className="text-neutral-400">GPIO Registers</div>
          <div className="space-y-1.5">
            {[
              { pin: 12, label: "Relay Pin 12" },
              { pin: 14, label: "Driver Pin 14" },
              { pin: 27, label: "Input Pin 27" },
            ].map(({ pin, label }) => {
              const isHigh = Number(gpio[String(pin)] ?? 0) === 1;
              return (
                <div key={pin} className="flex items-center justify-between py-1 border-b border-[#12141c]">
                  <span className="text-neutral-300">{label}</span>
                  <button
                    onClick={() =>
                      onTriggerAction(
                        isHigh ? `set GPIO pin ${pin} to low` : `set GPIO pin ${pin} to high`
                      )
                    }
                    disabled={isLoading}
                    className={`px-2 py-0.5 rounded text-[11px] transition-colors cursor-pointer border ${
                      isHigh
                        ? "bg-amber-500/10 text-amber-400 border-amber-500/25"
                        : "bg-[#10121a] text-neutral-500 border-[#1a1d28] hover:text-neutral-300"
                    }`}
                  >
                    {isHigh ? "HIGH (1)" : "LOW (0)"}
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* PWM Pin 18 */}
        <div className="p-3.5 rounded border border-[#141620] bg-[#07080c] space-y-2.5 font-mono text-xs">
          <div className="flex justify-between text-neutral-400">
            <span>PWM Pin 18</span>
            <span>{duty}% Duty · {pwm.freq}Hz</span>
          </div>

          <div className="w-full bg-[#12141c] h-1.5 rounded-full overflow-hidden my-3">
            <div
              className="bg-amber-400 h-full transition-all duration-200"
              style={{ width: `${duty}%` }}
            />
          </div>

          <div className="grid grid-cols-4 gap-1.5 pt-1">
            {[0, 25, 50, 75].map((val) => (
              <button
                key={val}
                onClick={() => onTriggerAction(`set pwm pin 18 duty cycle to ${val}`)}
                disabled={isLoading}
                className={`py-1 rounded text-[11px] cursor-pointer border transition-colors ${
                  duty === val
                    ? "bg-amber-500/15 text-amber-300 border-amber-500/30"
                    : "bg-[#10121a] text-neutral-400 hover:text-neutral-200 border-[#1a1d28]"
                }`}
              >
                {val}%
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
