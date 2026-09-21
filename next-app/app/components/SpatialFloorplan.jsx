"use client";

import React from "react";

export default function SpatialFloorplan({ state, onTriggerAction, isLoading }) {
  const s = state || {
    kitchen_lights: 100,
    living_lights: 70,
    bedroom_lights: 60,
    bathroom_lights: 0,
    thermostat: 20,
    front_door: "unlocked",
    back_door: "locked",
    blinds: "open",
    alarm: "off",
  };

  const isFrontLocked = s.front_door === "locked";
  const isBackLocked = s.back_door === "locked";
  const isBlindsOpen = s.blinds === "open";
  const isAlarmArmed = s.alarm === "armed" || s.alarm === "on";

  return (
    <div className="bg-[#0a0b10] border border-[#181a24] rounded-md p-5 space-y-4">
      <div className="flex items-center justify-between text-xs font-mono">
        <span className="text-neutral-200 font-medium">Physical State</span>
        <span className="text-neutral-500">Live Hardware Sync</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {/* Kitchen */}
        <div className="p-3.5 rounded border border-[#141620] bg-[#07080c] flex flex-col justify-between space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-neutral-300">Kitchen</span>
            <span className={s.kitchen_lights > 0 ? "text-amber-400" : "text-neutral-500"}>
              {s.kitchen_lights > 0 ? `${s.kitchen_lights}%` : "Off"}
            </span>
          </div>
          <div className="flex items-center space-x-1.5 font-mono text-xs">
            <button
              onClick={() => onTriggerAction(s.kitchen_lights > 0 ? "turn off kitchen lights" : "turn on kitchen lights")}
              disabled={isLoading}
              className="flex-1 py-1 px-2.5 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-300 hover:text-white border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              {s.kitchen_lights > 0 ? "Turn off" : "Turn on"}
            </button>
            <button
              onClick={() => onTriggerAction("dim kitchen lights to 50 percent")}
              disabled={isLoading}
              className="py-1 px-2 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-400 hover:text-neutral-200 border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              50%
            </button>
          </div>
        </div>

        {/* Living Room */}
        <div className="p-3.5 rounded border border-[#141620] bg-[#07080c] flex flex-col justify-between space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-neutral-300">Living Room</span>
            <span className={s.living_lights > 0 ? "text-amber-400" : "text-neutral-500"}>
              {s.living_lights > 0 ? `${s.living_lights}%` : "Off"}
            </span>
          </div>
          <div className="flex items-center space-x-1.5 font-mono text-xs">
            <button
              onClick={() => onTriggerAction(s.living_lights > 0 ? "turn off living room lights" : "turn on living room lights")}
              disabled={isLoading}
              className="flex-1 py-1 px-2.5 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-300 hover:text-white border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              {s.living_lights > 0 ? "Turn off" : "Turn on"}
            </button>
            <button
              onClick={() => onTriggerAction(isBlindsOpen ? "close blinds" : "open blinds")}
              disabled={isLoading}
              className="py-1 px-2.5 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-400 hover:text-neutral-200 border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              {isBlindsOpen ? "Blinds: open" : "Blinds: closed"}
            </button>
          </div>
        </div>

        {/* Bedroom */}
        <div className="p-3.5 rounded border border-[#141620] bg-[#07080c] flex flex-col justify-between space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-neutral-300">Bedroom</span>
            <span className={s.bedroom_lights > 0 ? "text-amber-400" : "text-neutral-500"}>
              {s.bedroom_lights > 0 ? `${s.bedroom_lights}%` : "Off"}
            </span>
          </div>
          <div className="flex items-center space-x-1.5 font-mono text-xs">
            <button
              onClick={() => onTriggerAction(s.bedroom_lights > 0 ? "turn off bedroom lights" : "turn on bedroom lights")}
              disabled={isLoading}
              className="flex-1 py-1 px-2.5 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-300 hover:text-white border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              {s.bedroom_lights > 0 ? "Turn off" : "Turn on"}
            </button>
            <button
              onClick={() => onTriggerAction("dim bedroom lights to 10 percent")}
              disabled={isLoading}
              className="py-1 px-2 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-400 hover:text-neutral-200 border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              10%
            </button>
          </div>
        </div>

        {/* Climate */}
        <div className="p-3.5 rounded border border-[#141620] bg-[#07080c] flex flex-col justify-between space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-neutral-300">Thermostat</span>
            <span className="text-neutral-100 font-semibold">{s.thermostat || 20}°C</span>
          </div>
          <div className="flex items-center space-x-1.5 font-mono text-xs">
            <button
              onClick={() => onTriggerAction(`set thermostat to ${Number(s.thermostat || 20) - 1}`)}
              disabled={isLoading}
              className="flex-1 py-1 px-2 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-300 hover:text-white border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              -1°C
            </button>
            <button
              onClick={() => onTriggerAction("set thermostat to 21")}
              disabled={isLoading}
              className="flex-1 py-1 px-2 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-300 hover:text-white border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              21°C
            </button>
            <button
              onClick={() => onTriggerAction(`set thermostat to ${Number(s.thermostat || 20) + 1}`)}
              disabled={isLoading}
              className="flex-1 py-1 px-2 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-300 hover:text-white border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              +1°C
            </button>
          </div>
        </div>

        {/* Doors */}
        <div className="p-3.5 rounded border border-[#141620] bg-[#07080c] flex flex-col justify-between space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-neutral-300">Doors</span>
            <span className="text-neutral-500">
              {isFrontLocked ? "Front locked" : "Front open"}
            </span>
          </div>
          <div className="flex items-center space-x-1.5 font-mono text-xs">
            <button
              onClick={() => onTriggerAction(isFrontLocked ? "unlock front door" : "lock front door")}
              disabled={isLoading}
              className="flex-1 py-1 px-2 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-300 hover:text-white border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              Front: {isFrontLocked ? "Unlock" : "Lock"}
            </button>
            <button
              onClick={() => onTriggerAction(isBackLocked ? "unlock back door" : "lock back door")}
              disabled={isLoading}
              className="flex-1 py-1 px-2 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-300 hover:text-white border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              Back: {isBackLocked ? "Unlock" : "Lock"}
            </button>
          </div>
        </div>

        {/* Security & Bathroom */}
        <div className="p-3.5 rounded border border-[#141620] bg-[#07080c] flex flex-col justify-between space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-neutral-300">Security</span>
            <span className={isAlarmArmed ? "text-red-400" : "text-neutral-500"}>
              {isAlarmArmed ? "Armed" : "Disarmed"}
            </span>
          </div>
          <div className="flex items-center space-x-1.5 font-mono text-xs">
            <button
              onClick={() => onTriggerAction(isAlarmArmed ? "disarm alarm" : "arm alarm")}
              disabled={isLoading}
              className="flex-1 py-1 px-2.5 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-300 hover:text-white border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              {isAlarmArmed ? "Disarm alarm" : "Arm alarm"}
            </button>
            <button
              onClick={() => onTriggerAction(s.bathroom_lights > 0 ? "turn off bathroom lights" : "turn on bathroom lights")}
              disabled={isLoading}
              className="py-1 px-2 rounded bg-[#11131a] hover:bg-[#181a24] text-neutral-400 hover:text-neutral-200 border border-[#1c1f2b] transition-colors cursor-pointer disabled:opacity-40"
            >
              Bath: {s.bathroom_lights > 0 ? "Off" : "On"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
