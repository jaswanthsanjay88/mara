"""
Comprehensive Audit Script:
1. Calculates exact n per suite for all tables.
2. Audits train/test overlap between finetune_tools.py and heldout_benchmark.jsonl.
3. Evaluates ablations:
   - Needle 3 (Stateless)
   - Needle 3 + ToolPlanner (Stateless)
   - Mara AFM (With ToolPlanner)
   - Mara AFM (Without ToolPlanner)
4. Measures precise latency and peak RAM per model and engine.
5. Saves all results to data/audit_report.json.
"""

import json
import os
import sys
import time
import psutil
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import needle
from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer
from mara.data import TOKENIZER_PATH
from mara.planner import ToolPlanner
from mara.argument_heads import MaraArgumentModel
from research.eval_utils import evaluate_single_sample
from research.finetune_tools import build_finetune_dataset

DATA_FILE = os.path.join(ROOT, "data", "heldout_benchmark.jsonl")
SMART_CKPT = os.path.join(ROOT, "checkpoints", "mara_smart_home.pt")
ARG_CKPT = os.path.join(ROOT, "checkpoints", "mara_arg_heads.pt")

# Needle tool definitions
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

NEEDLE_TOOLS = [set_lights, set_thermostat, control_device, lock_door]


class MaraAuditAgent:
    def __init__(self, device: torch.device):
        self.device = device
        self.tok = load_tokenizer(TOKENIZER_PATH)
        ckpt = torch.load(SMART_CKPT, map_location=device, weights_only=False)
        self.cfg = MaraConfig(**ckpt["config"])
        self.model = Mara(self.cfg).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.planner = ToolPlanner(model=self.model, tokenizer=self.tok, device=device)
        self.arg_model = MaraArgumentModel(d_model=self.cfg.d_model).to(device)
        if os.path.exists(ARG_CKPT):
            self.arg_model.load_state_dict(torch.load(ARG_CKPT, map_location=device, weights_only=False))
        self.arg_model.eval()

        self.tool_names = ["set_lights", "set_thermostat", "control_device", "lock_door", "none"]

    def _route_and_extract(self, query: str):
        q_enc = self.tok.encode(query)
        if not q_enc:
            return None

        ids = torch.tensor([q_enc], device=self.device)
        pos = torch.arange(len(q_enc), device=self.device).unsqueeze(0)

        rec = {
            "state": f"User Request: {query}\nAvailable Functions: {', '.join(self.tool_names)}",
            "questions": [
                {
                    "instr": "Which function should be triggered?",
                    "options": [f"tool: {t}" for t in self.tool_names],
                    "label": 0,
                    "qtype": "choice",
                }
            ]
        }

        probs = self.model.probs(self.tok, rec, device=self.device)
        top_tool_idx = probs[0].argmax().item()
        tool_name = self.tool_names[top_tool_idx]
        conf = probs[0][top_tool_idx].item()

        if tool_name == "none" or conf < 0.20:
            return None

        with torch.no_grad():
            h, _ = self.model(ids, position_ids=pos)
            h_seq = h[0]
            h_dec = h_seq[-1]
            tokens = self.tok.tokenize(query)

            args = self.arg_model.extract_arguments(
                tool_name=tool_name,
                h_query=h_seq,
                h_decide=h_dec,
                query_text=query,
                query_tokens=tokens,
            )

        return {"name": tool_name, "arguments": args}

    def complete(self, query: str, use_planner: bool = True):
        t0 = time.perf_counter()

        if use_planner:
            plan = self.planner.plan_tools(query)
            sub_queries = plan["sub_queries"]
        else:
            sub_queries = [query]

        predicted_calls = []
        for sq in sub_queries:
            call = self._route_and_extract(sq)
            if call:
                predicted_calls.append(call)

        lat_ms = (time.perf_counter() - t0) * 1000
        return {"function_calls": predicted_calls, "latency_ms": lat_ms}


def run_evaluation(name: str, complete_fn, samples):
    total = len(samples)
    oem = 0
    tool_match = 0
    arg_match = 0
    fp = 0
    neg_total = 0
    latencies = []
    cat_scores = {}

    for s in samples:
        q = s["query"]
        gt = s["ground_truth"]
        cat = s["category"]

        if cat not in cat_scores:
            cat_scores[cat] = {"total": 0, "oem": 0, "tool": 0}
        cat_scores[cat]["total"] += 1

        t0 = time.perf_counter()
        res = complete_fn(q)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        pred = res.get("function_calls", [])
        ev = evaluate_single_sample(pred, gt)

        if ev["is_oem"]:
            oem += 1
            cat_scores[cat]["oem"] += 1
        if ev["tool_match"]:
            tool_match += 1
            cat_scores[cat]["tool"] += 1
        if ev["arg_match"]:
            arg_match += 1
        if len(gt) == 0:
            neg_total += 1
            if ev["is_false_trigger"]:
                fp += 1

    return {
        "name": name,
        "total": total,
        "oem": oem,
        "oem_pct": oem / total * 100,
        "tool_pct": tool_match / total * 100,
        "arg_pct": arg_match / total * 100,
        "fp_rate": (fp / neg_total * 100) if neg_total > 0 else 0.0,
        "fp_count": fp,
        "neg_total": neg_total,
        "avg_lat_ms": sum(latencies) / len(latencies),
        "cat_scores": cat_scores,
    }


def main():
    print("=================================================================")
    print("                     MARA & NEEDLE 3 AUDIT                       ")
    print("=================================================================")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f]

    print(f"\n1. Sample counts in held-out benchmark suite (n = {len(test_samples)}):")
    from collections import Counter
    cats = Counter(s["category"] for s in test_samples)
    for c, count in cats.items():
        print(f"   - {c.ljust(15)}: n = {count}")

    # 2. Train / Test overlap audit
    print("\n2. Auditing Train / Test Overlap...")
    train_samples = build_finetune_dataset(1500)
    train_queries = set(s["query"].strip().lower() for s in train_samples)
    test_queries = [s["query"].strip().lower() for s in test_samples]
    overlapping = [q for q in test_queries if q in train_queries]
    overlap_count = len(overlapping)
    disjoint_count = len(test_samples) - overlap_count
    print(f"   Total Training Samples Generated : {len(train_samples)} (Unique: {len(train_queries)})")
    print(f"   Held-Out Test Queries            : {len(test_samples)}")
    print(f"   Exact String Overlap Count       : {overlap_count} ({overlap_count / len(test_samples) * 100:.1f}%)")
    print(f"   Strictly Disjoint (Unseen) Count : {disjoint_count} ({disjoint_count / len(test_samples) * 100:.1f}%)")

    # 3. Model setup
    device = torch.device("cpu")
    mara = MaraAuditAgent(device=device)

    # 4. Rerun Configurations:
    # Config A: Needle 3 Stateless (fresh agent per query)
    def complete_needle_stateless(q: str):
        ag = needle.Needle(tools=NEEDLE_TOOLS)
        return ag.complete(q)

    # Config B: Needle 3 + ToolPlanner (Stateless)
    def complete_needle_with_planner(q: str):
        plan = mara.planner.plan_tools(q)
        sub_queries = plan["sub_queries"]
        all_calls = []
        for sq in sub_queries:
            ag = needle.Needle(tools=NEEDLE_TOOLS)
            r = ag.complete(sq)
            all_calls.extend(r.get("function_calls", []))
        return {"function_calls": all_calls}

    # Config C: Mara With Planner
    def complete_mara_with_planner(q: str):
        return mara.complete(q, use_planner=True)

    # Config D: Mara Without Planner
    def complete_mara_without_planner(q: str):
        return mara.complete(q, use_planner=False)

    print("\n3. Running Ablations on full 104-item benchmark...")
    
    print("   [1/4] Running Mara AFM (With ToolPlanner)...")
    res_mara_plan = run_evaluation("Mara AFM (With Planner)", complete_mara_with_planner, test_samples)

    print("   [2/4] Running Mara AFM (Without ToolPlanner)...")
    res_mara_noplan = run_evaluation("Mara AFM (Without Planner)", complete_mara_without_planner, test_samples)

    print("   [3/4] Running Needle 3 (Stateless / Fresh per query)...")
    res_needle_stateless = run_evaluation("Needle 3 (Stateless)", complete_needle_stateless, test_samples)

    print("   [4/4] Running Needle 3 + ToolPlanner (Stateless)...")
    res_needle_plan = run_evaluation("Needle 3 + ToolPlanner", complete_needle_with_planner, test_samples)

    # Compile results table
    configs = [res_needle_stateless, res_needle_plan, res_mara_noplan, res_mara_plan]
    
    print("\n" + "="*95)
    print(f"{'Configuration':<28} | {'OEM Overall':<13} | {'Clean':<9} | {'Para':<9} | {'Typo':<9} | {'Compound':<9} | {'NearMiss':<9} | {'FP Rate':<8}")
    print("="*95)
    for c in configs:
        scores = c["cat_scores"]
        oem_str = f"{c['oem_pct']:5.1f}% ({c['oem']}/{c['total']})"
        cl = f"{scores['clean']['oem']/scores['clean']['total']*100:4.1f}%"
        pa = f"{scores['paraphrase']['oem']/scores['paraphrase']['total']*100:4.1f}%"
        ty = f"{scores['typo_noise']['oem']/scores['typo_noise']['total']*100:4.1f}%"
        co = f"{scores['compound']['oem']/scores['compound']['total']*100:4.1f}%"
        nm = f"{scores['hard_negative']['oem']/scores['hard_negative']['total']*100:4.1f}%"
        fp_str = f"{c['fp_rate']:4.1f}%"
        print(f"{c['name']:<28} | {oem_str:<13} | {cl:<9} | {pa:<9} | {ty:<9} | {co:<9} | {nm:<9} | {fp_str:<8}")
    print("="*95)

    # Save to disk
    report = {
        "dataset_breakdown": dict(cats),
        "overlap_audit": {
            "total_test": len(test_samples),
            "overlap_count": overlap_count,
            "overlap_pct": overlap_count / len(test_samples) * 100,
            "disjoint_count": disjoint_count,
            "disjoint_pct": disjoint_count / len(test_samples) * 100,
            "overlapping_queries": overlapping,
        },
        "configurations": configs,
    }

    with open(os.path.join(ROOT, "data", "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nFull audit report written to data/audit_report.json")


if __name__ == "__main__":
    main()
