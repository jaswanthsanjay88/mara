"use client";

import React, { useState } from "react";
import { ChevronDown } from "lucide-react";
import FloorPlan from "./FloorPlan";

export default function DevicePanel({
  smartHomeState,
  changedKeys = [],
  onReset,
  activeCalls = [],
  outcome = "refuse",
  hoveredCall = null,
}) {
  const [detailsOpen, setDetailsOpen] = useState(true);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between pb-1">
        <h2 className="text-[15px] leading-[22px] font-[500] text-[var(--fg)]">
          Device
        </h2>
        <span className="text-[12px] font-mono text-[var(--fg-3)]">
          Active Plan
        </span>
      </div>

      <div className="surface rounded-[10px] p-4 bg-[var(--fill-1)] border border-[var(--line)] space-y-4">
        <div className="space-y-4">
          {/* Top: 2D Floor Plan */}
          <div className="w-full">
            <FloorPlan
              state={smartHomeState}
              activeCalls={activeCalls}
              outcome={outcome}
              hoveredCall={hoveredCall}
            />
          </div>

          {/* Collapsible Details */}
          <div className="pt-2 border-t border-[var(--line)]">
            <button
              type="button"
              onClick={() => setDetailsOpen((prev) => !prev)}
              className="w-full flex items-center justify-between py-1.5 text-[13px] leading-[18px] font-[500] text-[var(--fg-2)] hover:text-[var(--fg)] transition-colors cursor-pointer"
              aria-expanded={detailsOpen}
            >
              <span>Details</span>
              <ChevronDown
                size={14}
                strokeWidth={1.5}
                className={`transform transition-transform duration-200 ${
                  detailsOpen ? "rotate-180" : ""
                }`}
              />
            </button>

            {detailsOpen && (
              <div className="mt-2 divide-y divide-[var(--line)]">
                {/* Rooms: Living room, Kitchen, Bedroom, Bathroom */}
                {smartHomeState?.rooms &&
                  Object.entries(smartHomeState.rooms).map(([roomName, info]) => {
                    const isChanged = changedKeys.includes(roomName);
                    const isOff = !info.on || info.brightness === 0;

                    return (
                      <div
                        key={roomName}
                        className={`h-[32px] px-2 flex items-center justify-between text-[13px] leading-[20px] rounded-[4px] transition-all duration-300 ${
                          isChanged
                            ? "bg-amber-500/15 text-amber-600 dark:text-amber-400 font-medium"
                            : ""
                        }`}
                      >
                        <span className="font-normal">{roomName}</span>
                        <div className="flex items-center space-x-2">
                          <span
                            className={`font-mono text-[12px] leading-[18px] px-1.5 py-0.5 rounded text-right tabular-nums ${
                              isOff
                                ? "text-[var(--fg-3)] bg-[var(--fill-2)]"
                                : "text-[var(--fg)] bg-[var(--fill-2)] font-medium"
                            }`}
                          >
                            {isOff ? "Off" : `${info.brightness}%`}
                          </span>
                        </div>
                      </div>
                    );
                  })}

                {/* Thermostat */}
                <div
                  className={`h-[32px] px-2 flex items-center justify-between text-[13px] leading-[20px] rounded-[4px] transition-all duration-300 ${
                    changedKeys.includes("thermostat")
                      ? "bg-amber-500/15 text-amber-600 dark:text-amber-400 font-medium"
                      : ""
                  }`}
                >
                  <span className="font-normal">Thermostat</span>
                  <span className="font-mono text-[12px] leading-[18px] px-1.5 py-0.5 rounded bg-[var(--fill-2)] text-[var(--fg)] tabular-nums font-medium">
                    {smartHomeState?.thermostat ?? 22}°C
                  </span>
                </div>

                {/* Fan and Garage door */}
                {smartHomeState?.devices &&
                  Object.entries(smartHomeState.devices).map(([devName, val]) => {
                    const isChanged = changedKeys.includes(devName);
                    const isActive = val === "On" || val === "Open";
                    return (
                      <div
                        key={devName}
                        className={`h-[32px] px-2 flex items-center justify-between text-[13px] leading-[20px] rounded-[4px] transition-all duration-300 ${
                          isChanged
                            ? "bg-amber-500/15 text-amber-600 dark:text-amber-400 font-medium"
                            : ""
                        }`}
                      >
                        <span className="font-normal">{devName}</span>
                        <span
                          className={`font-mono text-[12px] leading-[18px] px-1.5 py-0.5 rounded ${
                            isActive
                              ? "text-[var(--fg)] bg-[var(--fill-2)] font-medium"
                              : "text-[var(--fg-3)] bg-[var(--fill-2)]"
                          }`}
                        >
                          {val}
                        </span>
                      </div>
                    );
                  })}

                {/* Front door Lock */}
                {smartHomeState?.doors &&
                  Object.entries(smartHomeState.doors).map(([doorName, val]) => {
                    const isChanged = changedKeys.includes(doorName);
                    const isLocked = val === "Locked";
                    return (
                      <div
                        key={doorName}
                        className={`h-[32px] px-2 flex items-center justify-between text-[13px] leading-[20px] rounded-[4px] transition-all duration-300 ${
                          isChanged
                            ? "bg-amber-500/15 text-amber-600 dark:text-amber-400 font-medium"
                            : ""
                        }`}
                      >
                        <span className="font-normal">{doorName}</span>
                        <span
                          className={`font-mono text-[12px] leading-[18px] px-1.5 py-0.5 rounded ${
                            isLocked
                              ? "text-[var(--fg)] bg-[var(--fill-2)] font-medium"
                              : "text-amber-600 dark:text-amber-400 bg-amber-500/10 font-medium"
                          }`}
                        >
                          {val}
                        </span>
                      </div>
                    );
                  })}
              </div>
            )}
          </div>
        </div>

        {/* Reset Action */}
        <div className="pt-2 border-t border-[var(--line)]">
          <button
            type="button"
            onClick={onReset}
            className="text-[13px] leading-[18px] text-[var(--fg-2)] hover:text-[var(--fg)] transition-colors cursor-pointer"
          >
            Reset devices
          </button>
        </div>
      </div>
    </div>
  );
}
