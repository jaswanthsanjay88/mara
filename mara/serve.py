"""
Mara AFM Live Inference Server.
Serves the real PyTorch Automation Foundation Model weights (checkpoints/mara_afm.pt)
over a lightweight HTTP API and hosts the interactive visual sandbox.
"""

import json
import os
import re
import sys
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict
import urllib.parse

# Ensure repository root is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer, encode_record
from mara.afm import default_registry
from mara.planner import ToolPlanner

CKPT_PATH = os.path.join(ROOT_DIR, "checkpoints", "mara_afm.pt")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "data", "tokenizer.json")
DEMO_HTML_PATH = os.path.join(ROOT_DIR, "demo.html")

# Global device, neural network model and tool planner instances
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = None
MODEL = None
MODEL_CONFIG = None
PLANNER = None

# Live hardware & home state
LIVE_STATE = {
    "kitchen_lights": 100,
    "living_lights": 70,
    "bedroom_lights": 60,
    "bathroom_lights": 0,
    "thermostat": 20,
    "front_door": "unlocked",
    "back_door": "locked",
    "blinds": "open",
    "alarm": "off",
    "gpio": {14: 0, 12: 1, 27: 0},
    "pwm": {18: {"duty": 0, "freq": 1000}},
}


def load_model():
    global TOKENIZER, MODEL, MODEL_CONFIG, PLANNER
    print(f"[Mara AFM Server] Loading tokenizer from: {TOKENIZER_PATH}")
    TOKENIZER = load_tokenizer(TOKENIZER_PATH)

    print(f"[Mara AFM Server] Loading weights from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    MODEL_CONFIG = MaraConfig(**ckpt["config"])
    MODEL = Mara(MODEL_CONFIG).to(DEVICE)
    MODEL.load_state_dict(ckpt["model"])
    MODEL.eval()
    PLANNER = ToolPlanner(MODEL, TOKENIZER, DEVICE)
    print(f"[Mara AFM Server] Ready! Model params: {MODEL.num_params():,} on {DEVICE} with ToolPlanner")


def parse_arguments_for_tool(tool_name: str, query: str) -> Dict[str, Any]:
    """Extracts typed arguments for the predicted tool based on the user's natural language command."""
    q_lower = query.lower()
    args: Dict[str, Any] = {}

    if tool_name == "control_device":
        # Determine room
        if "kitchen" in q_lower:
            args["room"] = "kitchen"
        elif "living" in q_lower:
            args["room"] = "living room"
        elif "bedroom" in q_lower or "master" in q_lower:
            args["room"] = "bedroom"
        elif "bathroom" in q_lower:
            args["room"] = "bathroom"
        elif "all" in q_lower:
            args["room"] = "all"
        else:
            args["room"] = "living room"

        # Determine device
        if "light" in q_lower or "lights" in q_lower:
            args["device"] = "light"
        elif "fan" in q_lower:
            args["device"] = "fan"
        elif "ac" in q_lower or "cooler" in q_lower or "air conditioner" in q_lower:
            args["device"] = "ac"
        elif "blinds" in q_lower or "curtain" in q_lower:
            args["device"] = "blinds"
        elif "door" in q_lower or "lock" in q_lower:
            args["device"] = "lock"
        elif "thermostat" in q_lower or "temp" in q_lower:
            args["device"] = "thermostat"
        else:
            args["device"] = "light"

        # Determine action & level
        pct_match = re.search(r"(\d+)%", q_lower)
        temp_match = re.search(r"(\d+)\s*(?:degrees|deg|°|c)?", q_lower)

        if "off" in q_lower or "switch off" in q_lower or "shut down" in q_lower or "kill" in q_lower:
            args["action"] = "turn_off"
            args["level"] = 0
        elif "dim" in q_lower:
            args["action"] = "dim"
            args["level"] = int(pct_match.group(1)) if pct_match else 40
        elif pct_match:
            args["action"] = "dim"
            args["level"] = int(pct_match.group(1))
        elif "temp" in q_lower or "degrees" in q_lower or "thermostat" in q_lower:
            args["action"] = "set_temp"
            args["level"] = int(temp_match.group(1)) if temp_match else 22
        elif "lock" in q_lower and "unlock" not in q_lower:
            args["action"] = "lock"
        elif "unlock" in q_lower:
            args["action"] = "unlock"
        elif "close" in q_lower or "shut" in q_lower:
            args["action"] = "close"
        elif "open" in q_lower:
            args["action"] = "open"
        else:
            args["action"] = "turn_on"
            args["level"] = 100

    elif tool_name == "gpio_write":
        pin_match = re.search(r"pin\s*(\d+)|gpio\s*(\d+)", q_lower)
        pin = int(pin_match.group(1) or pin_match.group(2)) if pin_match else 14
        val = 1 if any(k in q_lower for k in ["high", " 1", "on", "enable", "set"]) and "low" not in q_lower and "0" not in q_lower else 0
        args["pin"] = pin
        args["value"] = val

    elif tool_name == "gpio_read":
        pin_match = re.search(r"pin\s*(\d+)|gpio\s*(\d+)", q_lower)
        args["pin"] = int(pin_match.group(1) or pin_match.group(2)) if pin_match else 14

    elif tool_name == "pwm_set":
        pin_match = re.search(r"pin\s*(\d+)|gpio\s*(\d+)", q_lower)
        args["pin"] = int(pin_match.group(1) or pin_match.group(2)) if pin_match else 18
        pct_match = re.search(r"(\d+)%", q_lower)
        args["duty_cycle"] = int(pct_match.group(1)) if pct_match else 50
        freq_match = re.search(r"(\d+)\s*(?:hz|khz)", q_lower)
        if freq_match:
            args["frequency"] = int(freq_match.group(1))
        else:
            args["frequency"] = 1000

    elif tool_name == "read_sensor":
        for s in ["temperature", "humidity", "motion", "co2", "lux"]:
            if s in q_lower:
                args["sensor_type"] = s
                break
        else:
            args["sensor_type"] = "temperature"

        for r in ["kitchen", "living room", "living", "bedroom", "bathroom", "garage"]:
            if r in q_lower:
                args["location"] = r
                break
        else:
            args["location"] = "living room"

    elif tool_name == "schedule_timer":
        time_match = re.search(r"(\d+)\s*(minute|min|second|sec)", q_lower)
        if time_match:
            n = int(time_match.group(1))
            args["duration_seconds"] = n * 60 if "min" in time_match.group(2) else n
        else:
            args["duration_seconds"] = 30
        args["command"] = query

    return args


def apply_execution_to_state(tool_name: str, args: Dict[str, Any], result: Any):
    """Reflects hardware and device operations onto the live sandbox state."""
    global LIVE_STATE
    if tool_name == "control_device":
        room = args.get("room", "")
        dev = args.get("device", "")
        act = args.get("action", "")
        lvl = args.get("level", 100)

        if dev == "light":
            if "kitchen" in room:
                LIVE_STATE["kitchen_lights"] = lvl if act != "turn_off" else 0
            elif "living" in room:
                LIVE_STATE["living_lights"] = lvl if act != "turn_off" else 0
            elif "bedroom" in room:
                LIVE_STATE["bedroom_lights"] = lvl if act != "turn_off" else 0
            elif "bathroom" in room:
                LIVE_STATE["bathroom_lights"] = lvl if act != "turn_off" else 0
            elif "all" in room:
                val = lvl if act != "turn_off" else 0
                LIVE_STATE["kitchen_lights"] = val
                LIVE_STATE["living_lights"] = val
                LIVE_STATE["bedroom_lights"] = val
                LIVE_STATE["bathroom_lights"] = val

        elif dev == "thermostat" or act == "set_temp":
            LIVE_STATE["thermostat"] = lvl

        elif dev == "lock":
            if "unlock" in act:
                LIVE_STATE["front_door"] = "unlocked"
            else:
                LIVE_STATE["front_door"] = "locked"

        elif dev == "blinds":
            LIVE_STATE["blinds"] = "closed" if act == "close" else "open"

    elif tool_name == "gpio_write":
        pin = args.get("pin", 14)
        val = args.get("value", 0)
        LIVE_STATE["gpio"][pin] = val

    elif tool_name == "pwm_set":
        pin = args.get("pin", 18)
        LIVE_STATE["pwm"][pin] = {
            "duty": args.get("duty_cycle", 50),
            "freq": args.get("frequency", 1000)
        }


def run_live_inference(query: str) -> Dict[str, Any]:
    """
    Executes actual PyTorch model inference on the query using the AFM Tool Planner:
    1. Formulates an execution plan (Single-step, Compound multi-tool, or Macro-routine).
    2. For each step, runs model.forward_decision() using trained PointerHead weights.
    3. Calculates exact millisecond inference latency per step.
    4. Routes to predicted tools and executes Python tool functions.
    5. Returns JSON with live model plan, per-step confidence, and updated state.
    """
    global PLANNER
    if PLANNER is None:
        PLANNER = ToolPlanner(MODEL, TOKENIZER, DEVICE)

    plan_spec = PLANNER.plan_tools(query)
    sub_queries = plan_spec["sub_queries"]
    reasons = plan_spec.get("reasons", [f"Step {i+1}" for i in range(len(sub_queries))])

    executed_steps = []
    total_inference_ms = 0.0
    tool_names = list(default_registry.tools.keys()) + ["none"]

    for idx, (sub_q, reason) in enumerate(zip(sub_queries, reasons), start=1):
        rec = default_registry.compile_fast_path_record(sub_q)
        packed = encode_record(TOKENIZER, rec)

        # Real neural model forward pass
        t0 = time.perf_counter()
        with torch.no_grad():
            probs, _ = MODEL.forward_decision(packed, device=DEVICE)
        step_ms = round((time.perf_counter() - t0) * 1000, 2)
        total_inference_ms += step_ms

        tool_probs = probs[0]
        best_idx = int(tool_probs.argmax().item())
        chosen_tool = tool_names[best_idx]
        confidence = round(float(tool_probs[best_idx].item()), 4)

        distribution = {
            name: round(float(tool_probs[i].item()), 4)
            for i, name in enumerate(tool_names)
        }

        if chosen_tool == "none":
            executed_steps.append({
                "step": idx,
                "sub_query": sub_q,
                "reason": reason,
                "tool": None,
                "confidence": confidence,
                "inference_latency_ms": step_ms,
                "status": "no_tool_required",
                "distribution": distribution,
                "execution": None,
            })
            continue

        args = parse_arguments_for_tool(chosen_tool, sub_q)
        exec_res = default_registry.execute({"name": chosen_tool, "arguments": args})
        apply_execution_to_state(chosen_tool, args, exec_res)

        executed_steps.append({
            "step": idx,
            "sub_query": sub_q,
            "reason": reason,
            "tool": chosen_tool,
            "confidence": confidence,
            "inference_latency_ms": step_ms,
            "arguments": args,
            "status": "executed",
            "distribution": distribution,
            "execution": exec_res,
        })

    is_multi_step = len(executed_steps) > 1 or plan_spec.get("is_macro", False)

    # If single step, maintain full backwards-compatibility attributes
    single_step = executed_steps[0] if executed_steps else {}
    first_tool = single_step.get("tool")
    first_conf = single_step.get("confidence", 1.0)

    if not is_multi_step and first_tool is None:
        status_str = "no_tool_required"
    elif is_multi_step:
        status_str = "planned_and_executed"
    else:
        status_str = "executed"

    return {
        "query": query,
        "status": status_str,
        "is_plan": is_multi_step,
        "tool": first_tool,
        "confidence": first_conf,
        "inference_latency_ms": round(total_inference_ms, 2),
        "arguments": single_step.get("arguments", {}),
        "distribution": single_step.get("distribution", {}),
        "execution": single_step.get("execution"),
        "plan": {
            "goal": plan_spec["goal"],
            "description": plan_spec.get("description", ""),
            "strategy": plan_spec.get("strategy", "sequential"),
            "total_steps": len(executed_steps),
            "steps": executed_steps,
        },
        "state": LIVE_STATE,
        "model_info": {
            "name": "Mara-AFM",
            "checkpoint": os.path.basename(CKPT_PATH),
            "parameters": MODEL.num_params(),
            "layers": MODEL_CONFIG.n_layers,
            "d_model": MODEL_CONFIG.d_model,
            "heads": MODEL_CONFIG.n_heads,
            "device": str(DEVICE),
        }
    }


class MaraHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Connection", "close")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(LIVE_STATE).encode("utf-8"))
            return

        if parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "model": "Mara-AFM",
                "params": MODEL.num_params() if MODEL else 0,
                "device": str(DEVICE),
            }).encode("utf-8"))
            return

        if parsed.path == "/" or parsed.path == "/index.html":
            if os.path.exists(DEMO_HTML_PATH):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(DEMO_HTML_PATH, "rb") as f:
                    self.wfile.write(f.read())
                return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"

        try:
            payload = json.loads(post_body) if post_body else {}
        except Exception:
            payload = {}

        if parsed.path == "/api/run":
            query = payload.get("query", "").strip()
            if not query:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Empty query"}).encode("utf-8"))
                return

            result = run_live_inference(query)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, indent=2).encode("utf-8"))
            return

        if parsed.path == "/api/reset":
            global LIVE_STATE
            LIVE_STATE = {
                "kitchen_lights": 100,
                "living_lights": 70,
                "bedroom_lights": 60,
                "bathroom_lights": 0,
                "thermostat": 20,
                "front_door": "unlocked",
                "back_door": "locked",
                "blinds": "open",
                "alarm": "off",
                "gpio": {14: 0, 12: 1, 27: 0},
                "pwm": {18: {"duty": 0, "freq": 1000}},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "reset", "state": LIVE_STATE}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def serve(port: int = 8000):
    load_model()
    server = ThreadingHTTPServer(("0.0.0.0", port), MaraHandler)
    print(f"\n========================================================")
    print(f" Mara AFM Neural Server Live on http://localhost:{port}")
    print(f" Endpoints: ")
    print(f"   GET  /            -> Interactive Sandbox UI")
    print(f"   POST /api/run     -> Live Neural Inference & Execution")
    print(f"   GET  /api/state   -> Real-time Hardware & Home State")
    print(f"   POST /api/reset   -> Reset State")
    print(f"========================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Mara AFM Server...")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.port)
