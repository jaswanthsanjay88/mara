/**
 * Mara Tool-Calling Playground Specification Implementation (Version 2)
 * Data contracts, Presets, Routing rule, and MockAdapter
 */

export const SMART_HOME_TOOLS = [
  {
    name: "set_lights",
    description: "Turn a room's lights on or off, or set their brightness from 0 to 100.",
    parameters: {
      room: { type: "string", required: true },
      on: { type: "boolean", required: true },
      brightness: { type: "integer", default: 100, required: false },
    },
  },
  {
    name: "set_thermostat",
    description: "Set the target temperature in degrees Celsius.",
    parameters: {
      temperature_c: { type: "number", required: true },
    },
  },
  {
    name: "control_device",
    description: "Switch or toggle a named device such as the fan or garage door.",
    parameters: {
      device: { type: "string", required: true },
      action: { type: "string", enum: ["on", "off", "toggle"], required: true },
    },
  },
  {
    name: "lock_door",
    description: "Lock or unlock a named door.",
    parameters: {
      door: { type: "string", required: true },
      locked: { type: "boolean", required: true },
    },
  },
];

export const PRESET_CONFIG = {
  "smart-home": {
    label: "Smart home",
    placeholder: "Tell the house what to do",
    defaultTools: SMART_HOME_TOOLS,
    chips: [
      "Dim the living room to 30",
      "Turn on the fan and lock the front door",
      "It's too dark in the bathroom",
      "What's the capital of France?",
    ],
  },
};

export const INITIAL_SMART_HOME_DEVICE_STATE = {
  rooms: {
    "Living room": { on: true, brightness: 60 },
    Kitchen: { on: true, brightness: 100 },
    Bedroom: { on: false, brightness: 0 },
    Bathroom: { on: false, brightness: 0 },
  },
  thermostat: 22,
  devices: {
    Fan: "Off",
    "Garage door": "Closed",
  },
  doors: {
    "Front door": "Locked",
  },
};

export const FLOOR = 0.1;

export function evaluateOutcome(response, actAt = 0.7) {
  if (!response) return "refuse";
  const calls = response.function_calls || [];
  const held = response.suppressed_calls || [];
  const confidence = response.confidence ?? 0;

  if (calls.length && confidence >= actAt) return "act";
  if (calls.length || held.length) return "confirm";
  return "refuse";
}

/**
 * MockAdapter matching MaraAdapter contract (name: "Mara", kind: "simulated")
 */
export class MockAdapter {
  name = "Mara";
  kind = "simulated";
  layers = 2;
  sizeMb = 2.7;

  async complete(prompt, tools) {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();

    // Artificial tiny async delay to emulate on-device engine turn (~55ms)
    await new Promise((resolve) => setTimeout(resolve, 55));

    const p = prompt.trim();
    const pLower = p.toLowerCase();
    const toolMap = new Map(tools.map((t) => [t.name, t]));

    // Canonical Smart Home presets matching Section 5 and Section 8
    if (pLower.includes("dim the living room to 30") || (pLower.includes("living room") && pLower.includes("30"))) {
      if (toolMap.has("set_lights")) {
        return this.createResponse({
          calls: [{ name: "set_lights", arguments: { room: "living room", on: true, brightness: 30 } }],
          reasoning: "'living room' -> room; 'dim' -> on true, brightness 30",
          confidence: 0.94,
          t0,
        });
      }
    }

    if (pLower.includes("turn on the fan and lock the front door") || (pLower.includes("fan") && pLower.includes("lock"))) {
      const calls = [];
      const reasons = [];
      if (toolMap.has("control_device")) {
        calls.push({ name: "control_device", arguments: { device: "fan", action: "on" } });
        reasons.push("'fan', 'turn on' -> control_device");
      }
      if (toolMap.has("lock_door")) {
        calls.push({ name: "lock_door", arguments: { door: "front door", locked: true } });
        reasons.push("'lock', 'front door' -> lock_door");
      }
      if (calls.length > 0) {
        return this.createResponse({
          calls,
          reasoning: reasons.join("; "),
          confidence: 0.91,
          t0,
        });
      }
    }

    if (pLower.includes("too dark in the bathroom") || (pLower.includes("dark") && pLower.includes("bathroom"))) {
      if (toolMap.has("set_lights")) {
        return this.createResponse({
          calls: [{ name: "set_lights", arguments: { room: "bathroom", on: true, brightness: 100 } }],
          reasoning: "'bathroom' -> room; 'dark' -> on true, brightness 100",
          confidence: 0.58, // Confirms (Ask first)
          t0,
        });
      }
    }

    if (pLower.includes("what's the capital of france") || pLower.includes("capital of france")) {
      return this.createEmptyResponse({
        reasoning: "No tool available for general factual questions",
        t0,
      });
    }

    // Dynamic extraction against active tools for custom input
    const calls = [];
    const reasons = [];

    // 1. set_lights
    if (toolMap.has("set_lights") && (pLower.includes("light") || pLower.includes("dim") || pLower.includes("bright") || pLower.includes("lamp") || pLower.includes("dark"))) {
      let room = "Living room";
      if (pLower.includes("bedroom")) room = "Bedroom";
      else if (pLower.includes("kitchen")) room = "Kitchen";
      else if (pLower.includes("bathroom") || pLower.includes("bath")) room = "Bathroom";

      const on = !pLower.includes("off");
      let brightness = on ? 100 : 0;
      const numMatch = pLower.match(/\b([0-9]{1,3})\b/);
      if (numMatch) {
        brightness = parseInt(numMatch[1], 10);
      } else if (pLower.includes("dim")) {
        brightness = 30;
      }
      calls.push({ name: "set_lights", arguments: { room, on, brightness } });
      reasons.push(`'${room.toLowerCase()}' -> room; brightness ${brightness}`);
    }

    // 2. set_thermostat
    if (toolMap.has("set_thermostat") && (pLower.includes("thermostat") || pLower.includes("temp") || pLower.includes("degree") || pLower.includes("celsius") || pLower.includes("heat") || pLower.includes("cool"))) {
      const numMatch = pLower.match(/([0-9]{1,2})/);
      const temp = numMatch ? parseFloat(numMatch[1]) : 21;
      calls.push({ name: "set_thermostat", arguments: { temperature_c: temp } });
      reasons.push(`'${temp}' -> temperature_c`);
    }

    // 3. control_device
    if (toolMap.has("control_device") && (pLower.includes("fan") || pLower.includes("garage") || pLower.includes("door"))) {
      let device = "fan";
      if (pLower.includes("garage")) device = "garage door";
      let action = "on";
      if (pLower.includes("off") || pLower.includes("close")) action = "off";
      else if (pLower.includes("toggle")) action = "toggle";
      calls.push({ name: "control_device", arguments: { device, action } });
      reasons.push(`'${device}' -> device; '${action}' -> action`);
    }

    // 4. lock_door
    if (toolMap.has("lock_door") && (pLower.includes("door") || pLower.includes("lock"))) {
      let door = "front door";
      if (pLower.includes("back")) door = "back door";
      const locked = !pLower.includes("unlock");
      calls.push({ name: "lock_door", arguments: { door, locked } });
      reasons.push(`'${door}' -> door; locked ${locked}`);
    }

    if (calls.length > 0) {
      let conf = 0.88;
      if (pLower.includes("maybe") || pLower.includes("could") || pLower.includes("think") || pLower.includes("dark") || pLower.includes("cold")) {
        conf = 0.55;
      }
      return this.createResponse({
        calls,
        reasoning: reasons.join("; "),
        confidence: conf,
        t0,
      });
    }

    return this.createEmptyResponse({
      reasoning: "No declared tool matched the request keywords or schema",
      t0,
    });
  }

  createResponse({ calls, reasoning, confidence, t0 }) {
    const tEnd = typeof performance !== "undefined" ? performance.now() : Date.now();
    const latency_ms = Math.max(1, Math.round(tEnd - t0));
    return {
      type: "call",
      success: true,
      error: null,
      error_code: null,
      function_calls: calls,
      suppressed_calls: confidence < FLOOR ? calls : [],
      reasoning: reasoning || "",
      confidence: Math.min(1, Math.max(0, confidence)),
      prefill_tps: 4300,
      decode_tps: 850,
      peak_ram_mb: 28.5,
      latency_ms,
    };
  }

  createEmptyResponse({ reasoning, t0 }) {
    const tEnd = typeof performance !== "undefined" ? performance.now() : Date.now();
    const latency_ms = Math.max(1, Math.round(tEnd - t0));
    return {
      type: "empty",
      success: true,
      error: null,
      error_code: null,
      function_calls: [],
      suppressed_calls: [],
      reasoning: reasoning || "",
      confidence: 0,
      prefill_tps: 4300,
      decode_tps: 850,
      peak_ram_mb: 28.5,
      latency_ms,
    };
  }
}
