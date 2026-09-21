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

export const COMPUTER_TOOLS = [
  {
    name: "open_url",
    description: "Open a web address in the browser.",
    parameters: {
      url: { type: "string", required: true },
    },
  },
  {
    name: "create_note",
    description: "Save a short note.",
    parameters: {
      text: { type: "string", required: true },
    },
  },
  {
    name: "start_timer",
    description: "Start a countdown timer.",
    parameters: {
      minutes: { type: "number", required: true },
    },
  },
  {
    name: "open_app",
    description: "Open an installed app by name.",
    parameters: {
      name: { type: "string", required: true },
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
  computer: {
    label: "Computer",
    placeholder: "Tell your computer what to do",
    defaultTools: COMPUTER_TOOLS,
    chips: [
      "Open example.com",
      "Note that the lease is due Friday",
      "Start a ten minute timer",
      "Book me a flight to Lisbon",
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

export const INITIAL_COMPUTER_LOG = [];

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

    // Canonical Computer presets matching Section 8
    if (pLower.includes("open example.com") || pLower.includes("example.com")) {
      if (toolMap.has("open_url")) {
        return this.createResponse({
          calls: [{ name: "open_url", arguments: { url: "example.com" } }],
          reasoning: "'example.com' -> url",
          confidence: 0.95,
          t0,
        });
      }
    }

    if (pLower.includes("note that the lease is due friday") || pLower.includes("lease is due friday")) {
      if (toolMap.has("create_note")) {
        return this.createResponse({
          calls: [{ name: "create_note", arguments: { text: "the lease is due Friday" } }],
          reasoning: "'lease is due Friday' -> text",
          confidence: 0.89,
          t0,
        });
      }
    }

    if (pLower.includes("start a ten minute timer") || pLower.includes("ten minute timer")) {
      if (toolMap.has("start_timer")) {
        return this.createResponse({
          calls: [{ name: "start_timer", arguments: { minutes: 10 } }],
          reasoning: "'ten minute' -> minutes 10",
          confidence: 0.93,
          t0,
        });
      }
    }

    if (pLower.includes("book me a flight to lisbon") || pLower.includes("flight to lisbon")) {
      return this.createEmptyResponse({
        reasoning: "No tool available for airline reservations",
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

    // 5. open_url
    if (toolMap.has("open_url") && (pLower.includes("http") || pLower.includes(".com") || pLower.includes(".org") || pLower.includes("url") || pLower.includes("site") || pLower.includes("web"))) {
      const urlMatch = pLower.match(/([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/);
      const url = urlMatch ? urlMatch[1] : "example.com";
      calls.push({ name: "open_url", arguments: { url } });
      reasons.push(`'${url}' -> url`);
    }

    // 6. create_note
    if (toolMap.has("create_note") && (pLower.includes("note") || pLower.includes("memo") || pLower.includes("remember") || pLower.includes("write down"))) {
      const text = p.replace(/^(create note|note that|note|write down|remember to|remember)\s*/i, "");
      calls.push({ name: "create_note", arguments: { text: text || p } });
      reasons.push(`'${text || p}' -> text`);
    }

    // 7. start_timer
    if (toolMap.has("start_timer") && (pLower.includes("timer") || pLower.includes("countdown") || pLower.includes("alarm"))) {
      const numMatch = pLower.match(/([0-9]+)/);
      const minutes = numMatch ? parseInt(numMatch[1], 10) : 5;
      calls.push({ name: "start_timer", arguments: { minutes } });
      reasons.push(`'${minutes}' -> minutes`);
    }

    // 8. open_app
    if (toolMap.has("open_app") && (pLower.includes("app") || pLower.includes("launch") || pLower.includes("open"))) {
      const nameMatch = p.replace(/^(launch|open app|open)\s*/i, "");
      calls.push({ name: "open_app", arguments: { name: nameMatch || "Terminal" } });
      reasons.push(`'${nameMatch || "Terminal"}' -> name`);
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
