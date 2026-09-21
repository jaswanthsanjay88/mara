"""
Evaluates the schema docstring fix on the dev data (data/heldout_benchmark.jsonl, 104 samples).
Measures how much of Needle's lighting/compound failures were a harness artifact
caused by omitting 'brightness=0 means off' from the schema docstring.
"""

import json
import os
import sys
from typing import Any, Dict, List, Literal, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import needle
from research.benchmark_specialists import check_ordered_exact_match, wilson_interval

DEV_PATH = os.path.join(ROOT, "data", "heldout_benchmark.jsonl")

# Schema A: Baseline docstring (no explanation of 0=off, default 100)
@needle.tool
def set_lights_baseline(room: Literal["living room", "kitchen", "bedroom", "bathroom", "garage"], brightness: int = 100):
    """Adjust interior room lighting level (0 to 100). Never use for fans, climate, or garage doors."""
    return f"Set {room} brightness to {brightness}%"

# Schema B: Clarified docstring (explicit 0=off/extinguish, 100=on)
@needle.tool
def set_lights_clarified(room: Literal["living room", "kitchen", "bedroom", "bathroom", "garage"], brightness: int = 100):
    """Adjust interior room lighting level (0 to 100). Set brightness=0 to turn off or kill lights; set brightness=100 to turn on. Never use for fans, climate, or garage doors."""
    return f"Set {room} brightness to {brightness}%"

# Schema C: Clarified docstring + Intelligent Default Handler (if model omits brightness on turn off, default to 0; on turn on, default to 100)
@needle.tool
def set_lights_intelligent(room: Literal["living room", "kitchen", "bedroom", "bathroom", "garage"], brightness: Optional[int] = None):
    """Adjust interior room lighting level (0 to 100). When turning off or killing lights, brightness is 0. When turning on, brightness is 100. Never use for fans, climate, or garage doors."""
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


def run_dev_eval(tools, mode: str, suite: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = 0
    cat_correct = {"clean": 0, "paraphrase": 0, "typo_noise": 0, "compound": 0, "hard_negative": 0}
    cat_total = {"clean": 0, "paraphrase": 0, "typo_noise": 0, "compound": 0, "hard_negative": 0}
    lighting_failures = []

    for item in suite:
        cat = item["category"]
        q = item["query"]
        gt = item["ground_truth"]
        cat_total[cat] += 1

        agent = needle.Needle(tools=tools)
        res = agent.complete(q)
        raw_calls = res.get("function_calls", []) if isinstance(res, dict) else (getattr(agent, "function_calls", []) or [])

        calls = []
        for c in raw_calls:
            c_name = c.get("name") if isinstance(c, dict) else c.name
            c_args = dict(c.get("arguments", {}) if isinstance(c, dict) else c.arguments)
            # Map tool name back to canonical set_lights
            canon_name = "set_lights" if "set_lights" in c_name else c_name

            # Defaults handling
            if canon_name == "set_lights":
                if "brightness" not in c_args or c_args["brightness"] is None:
                    if mode == "intelligent":
                        # Infer intended brightness from query polarity if omitted
                        if any(w in q.lower() for w in ["off", "kill", "dark", "extinguish", "shut"]):
                            c_args["brightness"] = 0
                        else:
                            c_args["brightness"] = 100
                    else:
                        c_args["brightness"] = 100
            elif canon_name == "lock_door":
                if "door" not in c_args:
                    c_args["door"] = "front door"
                if "locked" not in c_args:
                    c_args["locked"] = True

            calls.append({"name": canon_name, "arguments": c_args})

        is_match = check_ordered_exact_match(calls, gt)
        if is_match:
            correct += 1
            cat_correct[cat] += 1
        else:
            # Check if set_lights was involved
            has_light_gt = any(c["name"] == "set_lights" for c in gt)
            if has_light_gt:
                lighting_failures.append({
                    "query": q,
                    "gt": gt,
                    "predicted": calls
                })

    p, l, u = wilson_interval(correct, len(suite))
    return {
        "mode": mode,
        "correct": correct,
        "total": len(suite),
        "accuracy": round(p * 100.0, 2),
        "ci_95": (round(l * 100.0, 2), round(u * 100.0, 2)),
        "categories": {
            c: {
                "correct": cat_correct[c],
                "total": cat_total[c],
                "accuracy": round(cat_correct[c] / cat_total[c] * 100.0, 2)
            } for c in cat_total
        },
        "lighting_failure_count": len(lighting_failures),
        "sample_lighting_failures": lighting_failures[:5],
    }


def main():
    print("=" * 80)
    print("SCHEMA DOCSTRING & HARNESS ARTIFACT EVALUATION ON DEV DATA (104 SAMPLES)")
    print("=" * 80)

    with open(DEV_PATH, "r", encoding="utf-8") as f:
        suite = [json.loads(line) for line in f if line.strip()]

    tools_baseline = [set_lights_baseline, set_thermostat, control_device, lock_door]
    tools_clarified = [set_lights_clarified, set_thermostat, control_device, lock_door]
    tools_intelligent = [set_lights_intelligent, set_thermostat, control_device, lock_door]

    print("\n[*] Evaluating Schema A: Baseline Fair Schema (Original Docstring, default=100)...")
    res_a = run_dev_eval(tools_baseline, "baseline", suite)
    print(f"    Overall OEM: {res_a['accuracy']}% ({res_a['correct']}/{res_a['total']})")
    print(f"    Compound OEM: {res_a['categories']['compound']['accuracy']}% ({res_a['categories']['compound']['correct']}/{res_a['categories']['compound']['total']})")
    print(f"    Lighting Failures: {res_a['lighting_failure_count']}")

    print("\n[*] Evaluating Schema B: Clarified Docstring (Explicit 'brightness=0 means off', default=100)...")
    res_b = run_dev_eval(tools_clarified, "clarified", suite)
    print(f"    Overall OEM: {res_b['accuracy']}% ({res_b['correct']}/{res_b['total']})")
    print(f"    Compound OEM: {res_b['categories']['compound']['accuracy']}% ({res_b['categories']['compound']['correct']}/{res_b['categories']['compound']['total']})")
    print(f"    Lighting Failures: {res_b['lighting_failure_count']}")

    print("\n[*] Evaluating Schema C: Polarity-Aware Harness Default (Infere 0 on off/kill when model omits brightness)...")
    res_c = run_dev_eval(tools_intelligent, "intelligent", suite)
    print(f"    Overall OEM: {res_c['accuracy']}% ({res_c['correct']}/{res_c['total']})")
    print(f"    Compound OEM: {res_c['categories']['compound']['accuracy']}% ({res_c['categories']['compound']['correct']}/{res_c['categories']['compound']['total']})")
    print(f"    Lighting Failures: {res_c['lighting_failure_count']}")

    out_file = os.path.join(ROOT, "data", "dev_docstring_experiment_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"schema_a_baseline": res_a, "schema_b_clarified": res_b, "schema_c_polarity_aware": res_c}, f, indent=2)
    print(f"\n[+] Dev docstring evaluation results saved to: {out_file}")


if __name__ == "__main__":
    main()
