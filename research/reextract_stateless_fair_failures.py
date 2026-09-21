"""
Re-extracts Needle 3 failure traces under fair tool definitions and stateless execution.
Fair Tools:
- Literal types for all categorical slots.
- Explicit disclaimers preventing overlap between set_lights and control_device.
- Fresh Needle instance per query (stateless).
Evaluates on the original 104-item held-out benchmark (data/heldout_benchmark.jsonl).
"""

import json
import os
import sys
from typing import Literal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import needle
from research.eval_utils import evaluate_single_sample

DATA_FILE = os.path.join(ROOT, "data", "heldout_benchmark.jsonl")

# Fair Tool Schemas
@needle.tool
def set_lights(room: Literal["living room", "kitchen", "bedroom", "bathroom", "garage"], brightness: int = 100):
    """Adjust interior room lighting level (0 to 100). Never use for fans, climate, or garage doors."""
    return f"Set {room} brightness to {brightness}%"

@needle.tool
def set_thermostat(temperature_c: float):
    """Set the target climate temperature in degrees Celsius (16.0 to 28.0). Never use for lighting or doors."""
    return f"Thermostat target set to {temperature_c}C"

@needle.tool
def control_device(device: Literal["fan", "garage door"], action: Literal["on", "off", "open", "close"]):
    """Operate motor appliances strictly: ceiling fan (on/off) or garage door (open/close). Never use for room lights."""
    return f"{device} -> {action}"

@needle.tool
def lock_door(door: Literal["front door"] = "front door", locked: bool = True):
    """Engage or release the physical front door deadlock. Never use for garage door or lighting."""
    return f"Door {door} locked={locked}"

FAIR_TOOLS = [set_lights, set_thermostat, control_device, lock_door]


def main():
    print("Re-evaluating Needle 3 on 104 held-out benchmark with Fair Tools (Stateless)...")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    compound_failures = []
    near_miss_failures = []
    all_results = []

    for idx, s in enumerate(samples, 1):
        q = s["query"]
        gt = s["ground_truth"]
        cat = s["category"]

        # Strictly stateless instance per query
        agent = needle.Needle(tools=FAIR_TOOLS)
        res = agent.complete(q)
        pred = res.get("function_calls", [])
        ev = evaluate_single_sample(pred, gt)

        record = {
            "query": q,
            "ground_truth": gt,
            "pred_calls": pred,
            "confidence": res.get("confidence"),
            "reasoning": res.get("reasoning"),
            "is_oem": ev["is_oem"],
            "category": cat,
        }
        all_results.append(record)

        if not ev["is_oem"]:
            if cat == "compound":
                compound_failures.append(record)
            elif cat == "hard_negative":
                near_miss_failures.append(record)

    out_path = os.path.join(ROOT, "data", "stateless_fair_failures.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "compound_failures": compound_failures,
            "near_miss_failures": near_miss_failures,
            "all_results": all_results,
        }, f, indent=2)

    print(f"Stateless run finished. Compound failures: {len(compound_failures)}/20, Near-miss failures: {len(near_miss_failures)}/20")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
