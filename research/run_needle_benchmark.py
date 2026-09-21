"""
Needle 3 Evaluation Script on Held-Out Benchmark Suite.
Measures:
- Ordered Exact Match (OEM)
- Tool Selection Accuracy
- Argument Exact Match
- False Positive Trigger Rate
- Latency (ms) on CPU
- Peak RAM (MB)
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import needle
from research.eval_utils import evaluate_single_sample

DATA_DIR = os.path.join(ROOT, "data")
DATA_FILE = os.path.join(DATA_DIR, "heldout_benchmark.jsonl")

# 1. Define Needle tools
@needle.tool
def set_lights(room: str, brightness: int = 100):
    """Turn a room's lights on or off, or set their brightness from 0 to 100."""
    return f"Set {room} brightness to {brightness}%"

@needle.tool
def set_thermostat(temperature_c: float):
    """Set the target temperature in degrees Celsius."""
    return f"Thermostat target set to {temperature_c}C"

@needle.tool
def control_device(device: str, action: str):
    """Switch or toggle a named device like the fan or garage door."""
    return f"{device} -> {action}"

@needle.tool
def lock_door(door: str, locked: bool = True):
    """Lock or unlock a named door."""
    return f"Door {door} locked={locked}"


def run_needle_benchmark():
    agent = needle.Needle(tools=[set_lights, set_thermostat, control_device, lock_door])

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    total = len(samples)
    oem_count = 0
    tool_match_count = 0
    arg_match_count = 0
    false_trigger_count = 0
    negative_total = 0
    category_scores = {}
    latencies = []
    peak_rams = []

    print(f"Running Needle 3 on {total} held-out samples...")

    for i, item in enumerate(samples, start=1):
        q = item["query"]
        gt_calls = item["ground_truth"]
        cat = item["category"]

        if cat not in category_scores:
            category_scores[cat] = {"total": 0, "oem": 0, "tool": 0}
        category_scores[cat]["total"] += 1

        t0 = time.time()
        res = agent.complete(q)
        lat_ms = (time.time() - t0) * 1000
        latencies.append(lat_ms)

        if "peak_ram_mb" in res:
            peak_rams.append(res["peak_ram_mb"])

        pred_calls = res.get("function_calls", [])
        eval_res = evaluate_single_sample(pred_calls, gt_calls)

        if eval_res["is_oem"]:
            oem_count += 1
            category_scores[cat]["oem"] += 1

        if eval_res["tool_match"]:
            tool_match_count += 1
            category_scores[cat]["tool"] += 1

        if eval_res["arg_match"]:
            arg_match_count += 1

        if len(gt_calls) == 0:
            negative_total += 1
            if eval_res["is_false_trigger"]:
                false_trigger_count += 1

        if i % 20 == 0 or i == total:
            print(f"  Processed {i}/{total} samples (Current OEM: {oem_count/i*100:.1f}%)")

    avg_lat = sum(latencies) / len(latencies)
    avg_ram = sum(peak_rams) / len(peak_rams) if peak_rams else 100.0
    fp_rate = (false_trigger_count / negative_total * 100) if negative_total > 0 else 0.0

    print("\n=======================================================")
    print("           Needle 3 Benchmark Results                  ")
    print("=======================================================")
    print(f"Total Samples Evaluated  : {total}")
    print(f"Ordered Exact Match (OEM): {oem_count / total * 100:.1f}% ({oem_count}/{total})")
    print(f"Tool Selection Accuracy  : {tool_match_count / total * 100:.1f}% ({tool_match_count}/{total})")
    print(f"Argument Match Accuracy  : {arg_match_count / total * 100:.1f}% ({arg_match_count}/{total})")
    print(f"False Positive Trigger   : {fp_rate:.1f}% ({false_trigger_count}/{negative_total})")
    print(f"Avg Latency per Call     : {avg_lat:.1f} ms")
    print(f"Avg Peak RAM             : {avg_ram:.1f} MB")
    print("-------------------------------------------------------")
    print("Category Breakdown (OEM):")
    for c, stats in category_scores.items():
        pct = stats["oem"] / stats["total"] * 100
        print(f"  - {c.ljust(15)}: {pct:5.1f}% ({stats['oem']}/{stats['total']})")
    print("=======================================================")

    results = {
        "model": "Needle 3",
        "total": total,
        "oem_pct": oem_count / total * 100,
        "tool_pct": tool_match_count / total * 100,
        "arg_pct": arg_match_count / total * 100,
        "fp_rate": fp_rate,
        "avg_latency_ms": avg_lat,
        "peak_ram_mb": avg_ram,
        "category_scores": category_scores,
    }

    out_file = os.path.join(DATA_DIR, "needle3_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    run_needle_benchmark()
