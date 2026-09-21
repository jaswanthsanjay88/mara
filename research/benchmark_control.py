"""
Benchmark Needle 3 Control (needle3_control.cact, 63.44 MB zero-adapter archive)
against Needle 3 Specialist (needle3_specialist.cact, 63.44 MB) on the Frozen 250 Suite.
Also evaluates Mention-Order vs. Chronological Execution-Order on compound items.
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import needle
from research.reextract_stateless_fair_failures import FAIR_TOOLS
from research.benchmark_specialists import (
    check_ordered_exact_match,
    wilson_interval,
    NeedleEvaluator,
    run_benchmark_on_suite,
    TOOL_DEFAULTS
)

DATA_PATH = os.path.join(ROOT, "data", "frozen_eval_250.jsonl")
CONTROL_CACT = os.path.join(ROOT, "checkpoints", "needle3_control.cact")
SPECIALIST_CACT = os.path.join(ROOT, "checkpoints", "needle3_specialist.cact")


def get_chronological_ground_truth(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns ground truth in chronological execution order.
    Query 150: 'before you lock up the front door, kill the kitchen lights'
    Mention order: [lock_door, set_lights]
    Execution order: [set_lights, lock_door]
    """
    q = item["query"].lower()
    gt = item["ground_truth"]
    if "before you lock up the front door, kill the kitchen lights" in q:
        # Execution order is set_lights, then lock_door
        return [gt[1], gt[0]]
    return gt


def run_benchmark_with_both_orders(name: str, predict_fn, suite: List[Dict[str, Any]]) -> Dict[str, Any]:
    categories = ["clean", "paraphrase", "typo_noise", "compound", "hard_negative"]
    stats_mention = {c: {"total": 0, "correct": 0, "latencies": []} for c in categories}
    stats_exec = {c: {"total": 0, "correct": 0} for c in categories}
    
    overall_mention = 0
    overall_exec = 0
    all_latencies = []
    
    for idx, item in enumerate(suite):
        cat = item["category"]
        q = item["query"]
        gt_mention = item["ground_truth"]
        gt_exec = get_chronological_ground_truth(item)
        
        t0 = time.perf_counter()
        pred = predict_fn(q)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        all_latencies.append(lat_ms)
        
        match_mention = check_ordered_exact_match(pred, gt_mention)
        match_exec = check_ordered_exact_match(pred, gt_exec)
        
        stats_mention[cat]["total"] += 1
        stats_mention[cat]["latencies"].append(lat_ms)
        stats_exec[cat]["total"] += 1
        
        if match_mention:
            stats_mention[cat]["correct"] += 1
            overall_mention += 1
            
        if match_exec:
            stats_exec[cat]["correct"] += 1
            overall_exec += 1
            
        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx+1}/{len(suite)} queries...", flush=True)
            
    # Metrics
    mention_metrics = {}
    exec_metrics = {}
    for cat in categories:
        k_m = stats_mention[cat]["correct"]
        n = stats_mention[cat]["total"]
        p_m, ci_lm, ci_um = wilson_interval(k_m, n)
        mention_metrics[cat] = {
            "correct": k_m,
            "total": n,
            "accuracy": round(p_m * 100.0, 2),
            "ci_95": (round(ci_lm * 100.0, 2), round(ci_um * 100.0, 2)),
        }
        k_e = stats_exec[cat]["correct"]
        p_e, ci_le, ci_ue = wilson_interval(k_e, n)
        exec_metrics[cat] = {
            "correct": k_e,
            "total": n,
            "accuracy": round(p_e * 100.0, 2),
            "ci_95": (round(ci_le * 100.0, 2), round(ci_ue * 100.0, 2)),
        }
        
    ov_m_p, ov_m_l, ov_m_u = wilson_interval(overall_mention, len(suite))
    ov_e_p, ov_e_l, ov_e_u = wilson_interval(overall_exec, len(suite))
    
    hn_n = stats_mention["hard_negative"]["total"]
    hn_fp = hn_n - stats_mention["hard_negative"]["correct"]
    fp_rate, fp_l, fp_u = wilson_interval(hn_fp, hn_n)
    
    all_latencies.sort()
    return {
        "config": name,
        "mention_order": {
            "overall": {
                "correct": overall_mention,
                "total": len(suite),
                "accuracy": round(ov_m_p * 100.0, 2),
                "ci_95": (round(ov_m_l * 100.0, 2), round(ov_m_u * 100.0, 2)),
            },
            "per_category": mention_metrics,
        },
        "execution_order": {
            "overall": {
                "correct": overall_exec,
                "total": len(suite),
                "accuracy": round(ov_e_p * 100.0, 2),
                "ci_95": (round(ov_e_l * 100.0, 2), round(ov_e_u * 100.0, 2)),
            },
            "per_category": exec_metrics,
        },
        "false_positive_rate": {
            "fp_count": hn_fp,
            "total": hn_n,
            "rate": round(fp_rate * 100.0, 2),
            "ci_95": (round(fp_l * 100.0, 2), round(fp_u * 100.0, 2)),
        },
        "latency_p50_ms": round(all_latencies[len(all_latencies) // 2], 2),
        "latency_p95_ms": round(all_latencies[int(len(all_latencies) * 0.95)], 2),
    }


def main():
    print("=" * 80)
    print("EVALUATING TRUE CONTROL ARCHIVE (needle3_control.cact, 63.44 MB)")
    print("=" * 80)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        suite = [json.loads(l) for l in f if l.strip()]

    control_eval = NeedleEvaluator(weights=CONTROL_CACT, tau=0.50)
    spec_eval = NeedleEvaluator(weights=SPECIALIST_CACT, tau=0.50)

    configs = [
        ("Needle 3 Control (Zero-Adapter, No Planner)", lambda q: control_eval.predict(q, use_planner=False)),
        ("Needle 3 Control (Zero-Adapter + Planner)", lambda q: control_eval.predict(q, use_planner=True)),
        ("Needle 3 Specialist (LoRA, No Planner)", lambda q: spec_eval.predict(q, use_planner=False)),
        ("Needle 3 Specialist (LoRA + Planner)", lambda q: spec_eval.predict(q, use_planner=True)),
    ]

    results = {}
    for name, fn in configs:
        print(f"\n[*] Evaluating: {name}...", flush=True)
        res = run_benchmark_with_both_orders(name, fn, suite)
        results[name] = res
        ov_m = res["mention_order"]["overall"]
        ov_e = res["execution_order"]["overall"]
        fp = res["false_positive_rate"]
        print(f"    Mention-Order OEM:   {ov_m['accuracy']}% ({ov_m['correct']}/{ov_m['total']}) [CI: {ov_m['ci_95'][0]}% - {ov_m['ci_95'][1]}%]", flush=True)
        print(f"    Execution-Order OEM: {ov_e['accuracy']}% ({ov_e['correct']}/{ov_e['total']}) [CI: {ov_e['ci_95'][0]}% - {ov_e['ci_95'][1]}%]", flush=True)
        print(f"    FP Rate: {fp['rate']}% | Latency (p50): {res['latency_p50_ms']} ms", flush=True)

    out_file = os.path.join(ROOT, "data", "control_ablation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved control ablation results to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
