"use client";

import React, { useState } from "react";
import { ChevronDown } from "lucide-react";
import FloorPlan from "./FloorPlan";

export default function DevicePanel({
  presetId,
  smartHomeState,
  computerLog = [],
  changedKeys = [],
  onReset,
  activeCalls = [],
  outcome = "refuse",
  hoveredCall = null,
}) {
  const isSmartHome = presetId === "smart-home";
  const [detailsOpen, setDetailsOpen] = useState(true);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-[15px] leading-[22px] font-[500] text-[var(--fg)]">
          Device
        </h2>
      </div>

      <div className="surface rounded-[10px] p-4 bg-[var(--fill-1)] border border-[var(--line)] space-y-4">
        {isSmartHome ? (
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
                          className={`h-[32px] px-1 flex items-center justify-between text-[13px] leading-[20px] rounded-[4px] transition-colors ${
                            isChanged ? "animate-flash" : ""
                          }`}
                        >
                          <span className="text-[var(--fg)] font-normal">{roomName}</span>
                          <div className="flex items-center space-x-2">
                            {!isOff && (
                              <div className="w-[44px] h-[2px] bg-[var(--line-strong)] rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-[var(--fg)] transition-all"
                                  style={{ width: `${Math.min(100, info.brightness)}%` }}
                                />
                              </div>
                            )}
                            <span className="font-mono text-[12px] leading-[18px] text-[var(--fg-2)] w-7 text-right tabular-nums">
                              {isOff ? "Off" : info.brightness}
                            </span>
                          </div>
                        </div>
                      );
                    })}

                  {/* Thermostat */}
                  <div
                    className={`h-[32px] px-1 flex items-center justify-between text-[13px] leading-[20px] rounded-[4px] transition-colors ${
                      changedKeys.includes("thermostat") ? "animate-flash" : ""
                    }`}
                  >
                    <span className="text-[var(--fg)] font-normal">Thermostat</span>
                    <span className="font-mono text-[13px] leading-[20px] text-[var(--fg)] tabular-nums">
                      {smartHomeState?.thermostat ?? 22}°C
                    </span>
                  </div>

                  {/* Fan and Garage door */}
                  {smartHomeState?.devices &&
                    Object.entries(smartHomeState.devices).map(([devName, val]) => {
                      const isChanged = changedKeys.includes(devName);
                      return (
                        <div
                          key={devName}
                          className={`h-[32px] px-1 flex items-center justify-between text-[13px] leading-[20px] rounded-[4px] transition-colors ${
                            isChanged ? "animate-flash" : ""
                          }`}
                        >
                          <span className="text-[var(--fg)] font-normal">{devName}</span>
                          <span className="font-mono text-[12px] leading-[18px] text-[var(--fg-2)]">
                            {val}
                          </span>
                        </div>
                      );
                    })}

                  {/* Front door */}
                  {smartHomeState?.doors &&
                    Object.entries(smartHomeState.doors).map(([doorName, val]) => {
                      const isChanged = changedKeys.includes(doorName);
                      return (
                        <div
                          key={doorName}
                          className={`h-[32px] px-1 flex items-center justify-between text-[13px] leading-[20px] rounded-[4px] transition-colors ${
                            isChanged ? "animate-flash" : ""
                          }`}
                        >
                          <span className="text-[var(--fg)] font-normal">{doorName}</span>
                          <span className="font-mono text-[12px] leading-[18px] text-[var(--fg-2)]">
                            {val}
                          </span>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Computer Preset Log */
          <div className="space-y-2 font-mono text-[12px] leading-[18px]">
            {computerLog.length === 0 ? (
              <div className="text-[var(--fg-2)] py-4 font-normal">
                Nothing has run yet.
              </div>
            ) : (
              <div className="divide-y divide-[var(--line)]">
                {computerLog.slice(0, 5).map((entry, idx) => (
                  <div
                    key={idx}
                    className={`py-2 px-1 text-[var(--fg)] ${
                      idx === 0 && changedKeys.includes("computer_log") ? "animate-flash" : ""
                    }`}
                  >
                    {entry}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Reset Action */}
        <div className="pt-2 border-t border-[var(--line)]">
          <button
            type="button"
            onClick={onReset}
            className="text-[13px] leading-[18px] text-[var(--fg-2)] hover:text-[var(--fg)] transition-colors cursor-pointer"
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
