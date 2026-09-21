"use client";

import React, { useState, useEffect, useRef } from "react";

export default function FloorPlan({
  state,
  activeCalls = [],
  outcome = "refuse",
  hoveredCall = null,
  isInitialMount = true,
}) {
  const [playStepIndex, setPlayStepIndex] = useState(-1);
  const [activeBadges, setActiveBadges] = useState([]); // [{ stepNum, target, x, y }]
  const [activePings, setActivePings] = useState([]);   // [{ id, x, y }]
  const timerRefs = useRef([]);

  const rooms = state?.rooms || {
    "Living room": { on: true, brightness: 60 },
    Kitchen: { on: true, brightness: 100 },
    Bedroom: { on: false, brightness: 0 },
    Bathroom: { on: false, brightness: 0 },
  };

  const thermostat = state?.thermostat ?? 22;
  const isFanOn = state?.devices?.Fan === "On";
  const isGarageOpen = state?.devices?.["Garage door"] === "Open";
  const isFrontLocked = state?.doors?.["Front door"] === "Locked";

  // Target coordinates for rooms, devices, doors
  const targets = {
    "living room": { name: "Living room", type: "room", rect: { x: 10, y: 10, w: 180, h: 150 }, center: { x: 90, y: 48 } },
    "kitchen": { name: "Kitchen", type: "room", rect: { x: 190, y: 10, w: 120, h: 150 }, center: { x: 254, y: 52 } },
    "bedroom": { name: "Bedroom", type: "room", rect: { x: 10, y: 160, w: 100, h: 130 }, center: { x: 60, y: 180 } },
    "bathroom": { name: "Bathroom", type: "room", rect: { x: 150, y: 160, w: 60, h: 130 }, center: { x: 170, y: 200 } },
    "fan": { name: "Fan", type: "device", center: { x: 150, y: 50 }, radius: 14 },
    "garage door": { name: "Garage door", type: "device", center: { x: 260, y: 290 }, radius: 14 },
    "thermostat": { name: "Thermostat", type: "thermostat", center: { x: 130, y: 236 }, radius: 14 },
    "front door": { name: "Front door", type: "lock", center: { x: 130, y: 304 }, radius: 14 },
  };

  const getTargetForCall = (call) => {
    if (!call) return null;
    const name = call.name;
    const args = call.arguments || {};

    if (name === "set_lights") {
      const roomStr = String(args.room || "living").toLowerCase();
      if (roomStr.includes("kitchen")) return targets["kitchen"];
      if (roomStr.includes("bedroom") || roomStr.includes("bed")) return targets["bedroom"];
      if (roomStr.includes("bathroom") || roomStr.includes("bath")) return targets["bathroom"];
      return targets["living room"];
    }
    if (name === "set_thermostat") {
      return targets["thermostat"];
    }
    if (name === "control_device") {
      const devStr = String(args.device || "fan").toLowerCase();
      if (devStr.includes("garage")) return targets["garage door"];
      return targets["fan"];
    }
    if (name === "lock_door") {
      return targets["front door"];
    }
    return null;
  };

  // Play calls in sequence on Act outcome
  useEffect(() => {
    // Clear any previous timers
    timerRefs.current.forEach(clearTimeout);
    timerRefs.current = [];
    setActiveBadges([]);
    setActivePings([]);

    if (outcome !== "act" || !activeCalls || activeCalls.length === 0) {
      return;
    }

    activeCalls.forEach((call, k) => {
      const target = getTargetForCall(call);
      if (!target) return;

      const delay = k * 180;
      const timer = setTimeout(() => {
        const badgeItem = {
          stepNum: k + 1,
          x: target.center.x + 10,
          y: target.center.y - 10,
        };
        const pingItem = {
          id: `${k}-${Date.now()}`,
          x: target.center.x,
          y: target.center.y,
        };

        setActiveBadges((prev) => [...prev, badgeItem]);
        setActivePings((prev) => [...prev, pingItem]);

        // Remove ping after 650ms
        setTimeout(() => {
          setActivePings((prev) => prev.filter((p) => p.id !== pingItem.id));
        }, 650);

        // Remove badge after 1800ms
        setTimeout(() => {
          setActiveBadges((prev) => prev.filter((b) => b !== badgeItem));
        }, 1800);
      }, delay);

      timerRefs.current.push(timer);
    });

    return () => {
      timerRefs.current.forEach(clearTimeout);
    };
  }, [activeCalls, outcome]);

  // Thermostat arc calculation: 10°C to 30°C mapped to 0 to 270 degrees
  const clampedTemp = Math.max(10, Math.min(30, thermostat));
  const tempRatio = (clampedTemp - 10) / 20;
  const sweepDegrees = tempRatio * 270;
  const startAngle = 135; // Start at bottom-left
  const endAngle = startAngle + sweepDegrees;

  const polarToCartesian = (cx, cy, r, angleInDegrees) => {
    const angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180.0;
    return {
      x: cx + r * Math.cos(angleInRadians),
      y: cy + r * Math.sin(angleInRadians),
    };
  };

  const describeArc = (cx, cy, r, startAng, endAng) => {
    const start = polarToCartesian(cx, cy, r, endAng);
    const end = polarToCartesian(cx, cy, r, startAng);
    const largeArcFlag = endAng - startAng <= 180 ? "0" : "1";
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
  };

  // Accessibility label
  const ariaLabel = `Floor plan. Living room lights ${
    rooms["Living room"].on ? rooms["Living room"].brightness : "off"
  }. Kitchen lights ${
    rooms["Kitchen"].on ? rooms["Kitchen"].brightness : "off"
  }. Bedroom lights ${
    rooms["Bedroom"].on ? rooms["Bedroom"].brightness : "off"
  }. Bathroom lights ${
    rooms["Bathroom"].on ? rooms["Bathroom"].brightness : "off"
  }. Thermostat ${thermostat} degrees. Fan ${isFanOn ? "on" : "off"}. Garage door ${
    isGarageOpen ? "open" : "closed"
  }. Front door ${isFrontLocked ? "locked" : "unlocked"}.`;

  const hoveredTarget = hoveredCall ? getTargetForCall(hoveredCall) : null;
  const isConfirm = outcome === "confirm";

  return (
    <div className="w-full relative select-none">
      <svg
        role="img"
        aria-label={ariaLabel}
        viewBox="0 0 320 316"
        className="w-full h-auto overflow-visible"
        style={{ vectorEffect: "non-scaling-stroke" }}
      >
        <defs>
          {/* Light Tint Gradients with soft radial falloff */}
          {Object.entries(rooms).map(([roomName, info]) => {
            const b = info.on ? info.brightness / 100 : 0;
            const target = targets[roomName.toLowerCase()];
            if (!target) return null;
            const id = `tint-${roomName.toLowerCase().replace(/\s+/g, "-")}`;
            return (
              <radialGradient
                key={id}
                id={id}
                cx={target.center.x}
                cy={target.center.y}
                r={target.rect.w * 0.7}
                gradientUnits="userSpaceOnUse"
              >
                {/* Center: full strength */}
                <stop
                  offset="0%"
                  stopColor="var(--fg)"
                  stopOpacity={b * 0.22}
                />
                {/* Far edge: 45% of strength */}
                <stop
                  offset="100%"
                  stopColor="var(--fg)"
                  stopOpacity={b * 0.10}
                />
              </radialGradient>
            );
          })}
        </defs>

        {/* 1. Room Light Tints (Clipped to room rects) */}
        {Object.entries(rooms).map(([roomName, info]) => {
          const target = targets[roomName.toLowerCase()];
          if (!target) return null;
          const id = `tint-${roomName.toLowerCase().replace(/\s+/g, "-")}`;
          return (
            <rect
              key={`rect-tint-${roomName}`}
              x={target.rect.x}
              y={target.rect.y}
              width={target.rect.w}
              height={target.rect.h}
              fill={`url(#${id})`}
              className="transition-opacity duration-[var(--dur-slow)]"
            />
          );
        })}

        {/* 2. Furniture (quiet, 1px --line-strong, no fill) */}
        <g className={`fade-furniture ${isInitialMount ? "" : "opacity-100"}`}>
          {/* Living room sofa */}
          <rect
            x="34"
            y="96"
            width="72"
            height="26"
            rx="6"
            fill="none"
            stroke="var(--line-strong)"
            strokeWidth="1"
          />
          {/* Coffee table */}
          <circle
            cx="70"
            cy="75"
            r="10"
            fill="none"
            stroke="var(--line-strong)"
            strokeWidth="1"
          />
          {/* Kitchen counter along top */}
          <rect
            x="198"
            y="14"
            width="104"
            height="14"
            fill="none"
            stroke="var(--line-strong)"
            strokeWidth="1"
          />
          {/* Kitchen island */}
          <rect
            x="226"
            y="84"
            width="56"
            height="22"
            fill="none"
            stroke="var(--line-strong)"
            strokeWidth="1"
          />
          {/* Bedroom bed with two pillow rectangles */}
          <rect
            x="22"
            y="200"
            width="64"
            height="80"
            rx="4"
            fill="none"
            stroke="var(--line-strong)"
            strokeWidth="1"
          />
          <rect
            x="28"
            y="206"
            width="22"
            height="14"
            rx="2"
            fill="none"
            stroke="var(--line-strong)"
            strokeWidth="1"
          />
          <rect
            x="58"
            y="206"
            width="22"
            height="14"
            rx="2"
            fill="none"
            stroke="var(--line-strong)"
            strokeWidth="1"
          />
          {/* Bathroom tub */}
          <rect
            x="186"
            y="168"
            width="20"
            height="64"
            rx="3"
            fill="none"
            stroke="var(--line-strong)"
            strokeWidth="1"
          />
          {/* Bathroom toilet */}
          <circle
            cx="165"
            cy="260"
            r="7"
            fill="none"
            stroke="var(--line-strong)"
            strokeWidth="1"
          />
          {/* Garage car outline */}
          <rect
            x="234"
            y="190"
            width="52"
            height="88"
            rx="16"
            fill="none"
            stroke="var(--line)"
            strokeWidth="1"
            strokeDasharray="3 3"
          />
        </g>

        {/* 3. Exterior Walls (2px --fg) */}
        <path
          d="M 10 10 L 310 10 L 310 290 L 145 290 M 115 290 L 10 290 Z"
          fill="none"
          stroke="var(--fg)"
          strokeWidth="2"
          className={`draw-exterior ${isInitialMount ? "" : "stroke-dashoffset-0"}`}
        />

        {/* 4. Windows: two parallel 1px --fg-2 lines, 3px apart */}
        <g className={`fade-openings ${isInitialMount ? "" : "opacity-100"}`}>
          {/* Living room top window: x 40 to 100, y=10 */}
          <line x1="40" y1="8.5" x2="100" y2="8.5" stroke="var(--fg-2)" strokeWidth="1" />
          <line x1="40" y1="11.5" x2="100" y2="11.5" stroke="var(--fg-2)" strokeWidth="1" />

          {/* Living room left window: y 50 to 110, x=10 */}
          <line x1="8.5" y1="50" x2="8.5" y2="110" stroke="var(--fg-2)" strokeWidth="1" />
          <line x1="11.5" y1="50" x2="11.5" y2="110" stroke="var(--fg-2)" strokeWidth="1" />

          {/* Kitchen top window: x 220 to 280, y=10 */}
          <line x1="220" y1="8.5" x2="280" y2="8.5" stroke="var(--fg-2)" strokeWidth="1" />
          <line x1="220" y1="11.5" x2="280" y2="11.5" stroke="var(--fg-2)" strokeWidth="1" />

          {/* Bedroom left window: y 200 to 250, x=10 */}
          <line x1="8.5" y1="200" x2="8.5" y2="250" stroke="var(--fg-2)" strokeWidth="1" />
          <line x1="11.5" y1="200" x2="11.5" y2="250" stroke="var(--fg-2)" strokeWidth="1" />
        </g>

        {/* 5. Interior Walls (1.5px --fg) */}
        <g className={`draw-interior ${isInitialMount ? "" : "stroke-dashoffset-0"}`}>
          {/* Horizontal dividing wall (y=160): living/bed (10 to 115), hall gap (115 to 145), kitchen/garage (145 to 232 and 258 to 310) */}
          <line x1="10" y1="160" x2="115" y2="160" stroke="var(--fg)" strokeWidth="1.5" />
          <line x1="145" y1="160" x2="232" y2="160" stroke="var(--fg)" strokeWidth="1.5" />
          <line x1="258" y1="160" x2="310" y2="160" stroke="var(--fg)" strokeWidth="1.5" />

          {/* Living room / Kitchen vertical wall (x=190): y 10 to 50, gap 50 to 120, y 120 to 160 */}
          <line x1="190" y1="10" x2="190" y2="50" stroke="var(--fg)" strokeWidth="1.5" />
          <line x1="190" y1="120" x2="190" y2="160" stroke="var(--fg)" strokeWidth="1.5" />

          {/* Bedroom / Hall vertical wall (x=110): y 160 to 190, gap 190 to 216, y 216 to 290 */}
          <line x1="110" y1="160" x2="110" y2="190" stroke="var(--fg)" strokeWidth="1.5" />
          <line x1="110" y1="216" x2="110" y2="290" stroke="var(--fg)" strokeWidth="1.5" />

          {/* Hall / Bathroom vertical wall (x=150): y 160 to 190, gap 190 to 214, y 214 to 290 */}
          <line x1="150" y1="160" x2="150" y2="190" stroke="var(--fg)" strokeWidth="1.5" />
          <line x1="150" y1="214" x2="150" y2="290" stroke="var(--fg)" strokeWidth="1.5" />

          {/* Bathroom / Garage vertical wall (x=210): y 160 to 290 */}
          <line x1="210" y1="160" x2="210" y2="290" stroke="var(--fg)" strokeWidth="1.5" />
        </g>

        {/* 6. Door Swing Arcs (1px --fg-2) */}
        <g className={`fade-openings ${isInitialMount ? "" : "opacity-100"}`}>
          {/* Bedroom door: gap y 190 to 216, arc radius 26 into bedroom */}
          <path
            d="M 110 190 A 26 26 0 0 0 84 216"
            fill="none"
            stroke="var(--fg-2)"
            strokeWidth="1"
          />
          <line x1="110" y1="190" x2="84" y2="216" stroke="var(--fg-2)" strokeWidth="1" />

          {/* Bathroom door: gap y 190 to 214, arc radius 24 into bathroom */}
          <path
            d="M 150 190 A 24 24 0 0 1 174 214"
            fill="none"
            stroke="var(--fg-2)"
            strokeWidth="1"
          />
          <line x1="150" y1="190" x2="174" y2="214" stroke="var(--fg-2)" strokeWidth="1" />

          {/* Kitchen to garage door: gap x 232 to 258, arc radius 26 into garage */}
          <path
            d="M 232 160 A 26 26 0 0 0 258 186"
            fill="none"
            stroke="var(--fg-2)"
            strokeWidth="1"
          />
          <line x1="232" y1="160" x2="258" y2="186" stroke="var(--fg-2)" strokeWidth="1" />

          {/* Front door line across bottom gap (y=290, x 115 to 145) */}
          <line
            x1="115"
            y1="290"
            x2="145"
            y2="290"
            stroke="var(--fg)"
            strokeWidth="2"
          />
        </g>

        {/* 7. Garage Door (bottom wall, x 220 to 300) */}
        {isGarageOpen ? (
          /* Open: dashed 1px --fg-2 line 14px inside garage at y=276 */
          <line
            x1="220"
            y1="276"
            x2="300"
            y2="276"
            stroke="var(--fg-2)"
            strokeWidth="1"
            strokeDasharray="4 3"
            className="transition-opacity duration-300"
          />
        ) : (
          /* Closed: 3px --fg line with 4 short section ticks */
          <g className="transition-opacity duration-300">
            <line
              x1="220"
              y1="290"
              x2="300"
              y2="290"
              stroke="var(--fg)"
              strokeWidth="3"
            />
            <line x1="240" y1="288" x2="240" y2="292" stroke="var(--bg)" strokeWidth="1" />
            <line x1="260" y1="288" x2="260" y2="292" stroke="var(--bg)" strokeWidth="1" />
            <line x1="280" y1="288" x2="280" y2="292" stroke="var(--bg)" strokeWidth="1" />
          </g>
        )}

        {/* 8. Light Fixtures (circles of radius 4) */}
        {Object.entries(rooms).map(([roomName, info]) => {
          const target = targets[roomName.toLowerCase()];
          if (!target) return null;
          return (
            <circle
              key={`fixture-${roomName}`}
              cx={target.center.x}
              cy={target.center.y}
              r="4"
              fill={info.on ? "var(--fg)" : "none"}
              stroke={info.on ? "var(--fg)" : "var(--fg-2)"}
              strokeWidth="1"
              className="transition-colors duration-[var(--dur-base)]"
            />
          );
        })}

        {/* 9. Fan at (150, 50) */}
        <g id="fan-group">
          {/* Dashed sweep ring when fan is on */}
          {isFanOn && (
            <circle
              cx="150"
              cy="50"
              r="20"
              fill="none"
              stroke="var(--line-strong)"
              strokeWidth="1"
              strokeDasharray="3 3"
              className="transition-opacity duration-300"
            />
          )}

          {/* Rotating blades group */}
          <g
            className={isFanOn ? "fan-running" : ""}
            style={{ transformOrigin: "150px 50px" }}
          >
            {/* 4 blades at 90 deg intervals around (150, 50) */}
            <rect
              x="147"
              y="32"
              width="6"
              height="18"
              rx="3"
              fill={isFanOn ? "color-mix(in srgb, var(--fg) 24%, transparent)" : "none"}
              stroke={isFanOn ? "none" : "var(--fg-2)"}
              strokeWidth="1"
            />
            <rect
              x="150"
              y="47"
              width="18"
              height="6"
              rx="3"
              fill={isFanOn ? "color-mix(in srgb, var(--fg) 24%, transparent)" : "none"}
              stroke={isFanOn ? "none" : "var(--fg-2)"}
              strokeWidth="1"
            />
            <rect
              x="147"
              y="50"
              width="6"
              height="18"
              rx="3"
              fill={isFanOn ? "color-mix(in srgb, var(--fg) 24%, transparent)" : "none"}
              stroke={isFanOn ? "none" : "var(--fg-2)"}
              strokeWidth="1"
            />
            <rect
              x="132"
              y="47"
              width="18"
              height="6"
              rx="3"
              fill={isFanOn ? "color-mix(in srgb, var(--fg) 24%, transparent)" : "none"}
              stroke={isFanOn ? "none" : "var(--fg-2)"}
              strokeWidth="1"
            />
          </g>

          {/* Center Hub */}
          <circle cx="150" cy="50" r="3" fill="var(--fg)" />
        </g>

        {/* 10. Thermostat Dial at (130, 236), radius 12 */}
        <g id="thermostat-group">
          {/* Base outer ring */}
          <circle
            cx="130"
            cy="236"
            r="12"
            fill="var(--bg)"
            stroke="var(--line-strong)"
            strokeWidth="1"
          />

          {/* Sweeping temperature arc up to 270 deg */}
          {sweepDegrees > 0 && (
            <path
              d={describeArc(130, 236, 12, startAngle, endAngle)}
              fill="none"
              stroke="var(--fg)"
              strokeWidth="2"
              className="transition-all duration-[var(--dur-slow)]"
            />
          )}

          {/* Temperature Number in Mono 10 */}
          <text
            x="130"
            y="239.5"
            textAnchor="middle"
            fill="var(--fg)"
            className="font-mono text-[10px] font-normal tabular-nums select-none"
          >
            {Math.round(thermostat)}
          </text>
        </g>

        {/* 11. Front Door Lock Padlock Glyph at (130, 304) */}
        <g id="padlock-group" className="transition-all duration-200">
          {isFrontLocked ? (
            /* Locked: closed shackle, body filled --fg */
            <g transform="translate(124, 298)">
              {/* Shackle */}
              <path
                d="M 3 6 L 3 3.5 A 3 3 0 0 1 9 3.5 L 9 6"
                fill="none"
                stroke="var(--fg)"
                strokeWidth="1.5"
              />
              {/* Body */}
              <rect x="1" y="6" width="10" height="7" rx="1.5" fill="var(--fg)" />
            </g>
          ) : (
            /* Unlocked: shackle swung open ~30 deg, body outlined only */
            <g transform="translate(124, 298)">
              {/* Open Shackle */}
              <path
                d="M 3 6 L 3 3.5 A 3 3 0 0 1 9 3.5 L 11 1.5"
                fill="none"
                stroke="var(--fg)"
                strokeWidth="1.5"
              />
              {/* Outlined Body */}
              <rect
                x="1"
                y="6"
                width="10"
                height="7"
                rx="1.5"
                fill="none"
                stroke="var(--fg)"
                strokeWidth="1.5"
              />
            </g>
          )}
        </g>

        {/* 12. Room Labels and Readouts */}
        <g className="font-sans text-[12px] font-normal select-none pointer-events-none">
          {/* Living room */}
          <text x="18" y="24" fill="var(--fg-2)">
            Living room
          </text>
          {rooms["Living room"].on && (
            <text x="88" y="24" fill="var(--fg)" className="font-mono text-[12px]">
              {rooms["Living room"].brightness}
            </text>
          )}

          {/* Kitchen */}
          <text x="198" y="24" fill="var(--fg-2)">
            Kitchen
          </text>
          {rooms["Kitchen"].on && (
            <text x="246" y="24" fill="var(--fg)" className="font-mono text-[12px]">
              {rooms["Kitchen"].brightness}
            </text>
          )}

          {/* Bedroom */}
          <text x="18" y="174" fill="var(--fg-2)">
            Bedroom
          </text>
          {rooms["Bedroom"].on && (
            <text x="74" y="174" fill="var(--fg)" className="font-mono text-[12px]">
              {rooms["Bedroom"].brightness}
            </text>
          )}

          {/* Bathroom */}
          <text x="158" y="174" fill="var(--fg-2)">
            Bathroom
          </text>
          {rooms["Bathroom"].on && (
            <text x="202" y="174" fill="var(--fg)" className="font-mono text-[12px]">
              {rooms["Bathroom"].brightness}
            </text>
          )}

          {/* Garage */}
          <text x="218" y="174" fill="var(--fg-2)">
            {isGarageOpen ? "Garage  Open" : "Garage"}
          </text>
        </g>

        {/* 13. Ghost Previews on Confirm Outcome */}
        {isConfirm &&
          activeCalls.map((call, idx) => {
            const target = getTargetForCall(call);
            if (!target) return null;
            return (
              <g key={`ghost-${idx}`}>
                {/* Dashed room rect outline for lights */}
                {target.type === "room" && (
                  <rect
                    x={target.rect.x + 2}
                    y={target.rect.y + 2}
                    width={target.rect.w - 4}
                    height={target.rect.h - 4}
                    fill="none"
                    stroke="var(--fg)"
                    strokeWidth="1"
                    strokeDasharray="4 3"
                    className="opacity-70"
                  />
                )}
                {/* Dashed ring radius 14 around target */}
                <circle
                  cx={target.center.x}
                  cy={target.center.y}
                  r="14"
                  fill="none"
                  stroke="var(--fg)"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                  className="opacity-70"
                />
                {/* Outlined badge: 1px --fg circle, transparent fill, --fg numeral */}
                <g transform={`translate(${target.center.x + 10}, ${target.center.y - 10})`}>
                  <circle
                    cx="0"
                    cy="0"
                    r="8"
                    fill="var(--bg)"
                    stroke="var(--fg)"
                    strokeWidth="1"
                  />
                  <text
                    x="0"
                    y="3.5"
                    textAnchor="middle"
                    fill="var(--fg)"
                    className="font-mono text-[10px] font-normal"
                  >
                    {idx + 1}
                  </text>
                </g>
              </g>
            );
          })}

        {/* 14. Hover Highlight (Call to plan link) */}
        {hoveredTarget && (
          <g>
            {hoveredTarget.type === "room" ? (
              <rect
                x={hoveredTarget.rect.x}
                y={hoveredTarget.rect.y}
                width={hoveredTarget.rect.w}
                height={hoveredTarget.rect.h}
                fill="none"
                stroke="var(--fg)"
                strokeWidth="1.5"
                className="transition-opacity duration-[var(--dur-fast)]"
              />
            ) : (
              <circle
                cx={hoveredTarget.center.x}
                cy={hoveredTarget.center.y}
                r="14"
                fill="none"
                stroke="var(--fg)"
                strokeWidth="1.5"
                className="transition-opacity duration-[var(--dur-fast)]"
              />
            )}
          </g>
        )}

        {/* 15. Animated Ping Rings */}
        {activePings.map((ping) => (
          <circle
            key={ping.id}
            cx={ping.x}
            cy={ping.y}
            r="6"
            fill="none"
            stroke="var(--fg)"
            strokeWidth="1"
            className="ping-ring pointer-events-none"
          />
        ))}

        {/* 16. Active Order Badges (Act outcome) */}
        {activeBadges.map((badge, bIdx) => (
          <g
            key={`badge-${bIdx}`}
            transform={`translate(${badge.x}, ${badge.y})`}
            className="badge-animated pointer-events-none"
          >
            <circle cx="0" cy="0" r="8" fill="var(--fg)" />
            <text
              x="0"
              y="3.5"
              textAnchor="middle"
              fill="var(--bg)"
              className="font-mono text-[10px] font-normal"
            >
              {badge.stepNum}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
