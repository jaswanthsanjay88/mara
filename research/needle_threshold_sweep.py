"""
Confidence Threshold Sweep on the Old 104 Held-Out Benchmark.
Tests thresholds tau in [0.00, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95].
Gating Rule:
- If conf < tau: refuse execution (predict []).
- Evaluates impact on OEM, Tool Accuracy, and False Positive Trigger Rate.
Picks optimal tau* to freeze for the new independent evaluation suite.
"""

import json
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research.eval_utils import evaluate_single_sample
from research.reextract_stateless_fair_failures import FAIR_TOOLS
from research.run_audit_ablations import MaraAuditAgent

DATA_FILE = os.path.join(ROOT, "data", "heldout_benchmark.jsonl")
STATELESS_RESULTS = os.path.join(ROOT, "data", "stateless_fair_failures.json")


def sweep_needle():
    with open(STATELESS_RESULTS, "r", encoding="utf-8") as f:
        records = json.load(f)["all_results"]

    thresholds = [0.00, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    print("\n--- Needle 3 (Fair Tools, Stateless) Threshold Sweep on Old 104 ---")
    print(f"{'Threshold (tau)':<16} | {'OEM':<12} | {'FP Rate':<10} | {'Clean OEM':<10} | {'Compound OEM':<12}")
    print("-" * 68)

    best_tau = 0.0
    best_oem = 0.0

    for tau in thresholds:
        oem = 0
        fp = 0
        neg_count = 0
        clean_oem = 0
        comp_oem = 0

        for r in records:
            gt = r["ground_truth"]
            cat = r["category"]
            pred = r["pred_calls"]
            conf = r.get("confidence") or 1.0

            # Gating: if conf < tau, refuse call
            gated_pred = pred if conf >= tau else []
            ev = evaluate_single_sample(gated_pred, gt)

            if ev["is_oem"]:
                oem += 1
                if cat == "clean": clean_oem += 1
                if cat == "compound": comp_oem += 1
            if len(gt) == 0:
                neg_count += 1
                if ev["is_false_trigger"]:
                    fp += 1

        oem_pct = oem / len(records) * 100
        fp_rate = fp / neg_count * 100
        print(f"{tau:<16.2f} | {oem_pct:5.1f}% ({oem}/{len(records)}) | {fp_rate:4.1f}% ({fp}/{neg_count}) | {clean_oem/20*100:4.1f}%     | {comp_oem/20*100:4.1f}%")

        if oem_pct > best_oem:
            best_oem = oem_pct
            best_tau = tau

    print(f"Optimal Threshold for Needle 3: tau* = {best_tau:.2f} (OEM = {best_oem:.1f}%)")
    return best_tau


def sweep_mara():
    device = torch.device("cpu")
    mara = MaraAuditAgent(device=device)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    # Pre-compute Mara raw predictions and router confidences
    records = []
    for s in samples:
        q = s["query"]
        gt = s["ground_truth"]
        cat = s["category"]

        # Run complete pipeline to get calls and min_confidence across steps
        plan = mara.planner.plan_tools(q)
        sub_queries = plan["sub_queries"]
        step_calls = []
        min_conf = 1.0

        for sq in sub_queries:
            q_enc = mara.tok.encode(sq)
            if not q_enc:
                continue
            rec = {
                "state": f"User Request: {sq}\nAvailable Functions: {', '.join(mara.tool_names)}",
                "questions": [{"instr": "Which function?", "options": [f"tool: {t}" for t in mara.tool_names], "label": 0, "qtype": "choice"}]
            }
            probs = mara.model.probs(mara.tok, rec, device=device)
            top_idx = probs[0].argmax().item()
            c = probs[0][top_idx].item()
            tool = mara.tool_names[top_idx]
            min_conf = min(min_conf, c)

            call = mara._route_and_extract(sq)
            if call:
                step_calls.append(call)

        records.append({
            "query": q,
            "ground_truth": gt,
            "category": cat,
            "pred_calls": step_calls,
            "confidence": min_conf,
        })

    thresholds = [0.00, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    print("\n--- Mara AFM (With Planner) Threshold Sweep on Old 104 ---")
    print(f"{'Threshold (tau)':<16} | {'OEM':<12} | {'FP Rate':<10} | {'Clean OEM':<10} | {'Compound OEM':<12}")
    print("-" * 68)

    best_tau = 0.0
    best_oem = 0.0

    for tau in thresholds:
        oem = 0
        fp = 0
        neg_count = 0
        clean_oem = 0
        comp_oem = 0

        for r in records:
            gt = r["ground_truth"]
            cat = r["category"]
            pred = r["pred_calls"]
            conf = r["confidence"]

            gated_pred = pred if conf >= tau else []
            ev = evaluate_single_sample(gated_pred, gt)

            if ev["is_oem"]:
                oem += 1
                if cat == "clean": clean_oem += 1
                if cat == "compound": comp_oem += 1
            if len(gt) == 0:
                neg_count += 1
                if ev["is_false_trigger"]:
                    fp += 1

        oem_pct = oem / len(records) * 100
        fp_rate = fp / neg_count * 100
        print(f"{tau:<16.2f} | {oem_pct:5.1f}% ({oem}/{len(records)}) | {fp_rate:4.1f}% ({fp}/{neg_count}) | {clean_oem/20*100:4.1f}%     | {comp_oem/20*100:4.1f}%")

        if oem_pct > best_oem:
            best_oem = oem_pct
            best_tau = tau

    print(f"Optimal Threshold for Mara AFM: tau* = {best_tau:.2f} (OEM = {best_oem:.1f}%)")
    return best_tau


if __name__ == "__main__":
    sweep_needle()
    sweep_mara()
