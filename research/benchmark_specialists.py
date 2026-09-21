"""
Comprehensive 6-Way Ablation Benchmark on Frozen 250-Sample Evaluation Suite.
Evaluates:
  1. Mara AFM alone (without planner)
  2. Mara AFM + Planner
  3. Needle 3 Base (stateless, fair tools, no planner)
  4. Needle 3 Base + Planner
  5. Needle 3 Fine-Tuned Specialist (no planner)
  6. Needle 3 Fine-Tuned Specialist + Planner

Computes per-suite accuracy, overall accuracy, Wilson 95% confidence intervals,
and False Positive Trigger Rates on Hard Negatives.
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer
from mara.data import TOKENIZER_PATH
from mara.argument_heads import MaraArgumentModel
from mara.planner import ToolPlanner
from research.reextract_stateless_fair_failures import FAIR_TOOLS

import needle

DATA_PATH = os.path.join(ROOT, "data", "frozen_eval_250.jsonl")
SMART_CKPT = os.path.join(ROOT, "checkpoints", "mara_smart_home.pt")
CKPT_PATH = SMART_CKPT if os.path.exists(SMART_CKPT) else os.path.join(ROOT, "checkpoints", "mara_afm.pt")
ARG_CKPT_PATH = os.path.join(ROOT, "checkpoints", "mara_arg_heads.pt")


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Computes Wilson score 95% confidence interval for proportion k/n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.959963984540054  # 95% confidence
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    ci_low = max(0.0, center - half)
    ci_high = min(1.0, center + half)
    return p, ci_low, ci_high


def match_arguments(pred_args: Dict[str, Any], gt_args: Dict[str, Any]) -> bool:
    """Compares argument dictionaries with type tolerance for numbers."""
    if set(pred_args.keys()) != set(gt_args.keys()):
        return False
    for k, v_gt in gt_args.items():
        v_pred = pred_args[k]
        if isinstance(v_gt, (int, float)) and isinstance(v_pred, (int, float)):
            if abs(v_pred - v_gt) > 0.1:
                return False
        elif isinstance(v_gt, bool) and isinstance(v_pred, bool):
            if v_pred != v_gt:
                return False
        elif isinstance(v_gt, str) and isinstance(v_pred, str):
            if v_pred.strip().lower() != v_gt.strip().lower():
                return False
        else:
            if v_pred != v_gt:
                return False
    return True


TOOL_DEFAULTS = {
    "set_lights": {"brightness": 100},
    "lock_door": {"door": "front door", "locked": True},
}


def check_ordered_exact_match(preds: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> bool:
    """Ordered Exact Match (OEM) across full sequence of function calls and arguments."""
    if len(preds) != len(ground_truth):
        return False
    for p, gt in zip(preds, ground_truth):
        tool_name = gt.get("name")
        if p.get("name") != tool_name:
            return False
        
        # Populate defaults for missing arguments in prediction
        p_args = dict(p.get("arguments", {}))
        for def_k, def_v in TOOL_DEFAULTS.get(tool_name, {}).items():
            if def_k not in p_args:
                p_args[def_k] = def_v

        if not match_arguments(p_args, gt.get("arguments", {})):
            return False
    return True


class MaraEvaluator:
    def __init__(self, tau: float = 0.50):
        self.device = torch.device("cpu")
        self.tok = load_tokenizer(TOKENIZER_PATH)
        self.tau = tau

        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        self.cfg = MaraConfig(**ckpt["config"])
        self.model = Mara(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.planner = ToolPlanner(model=self.model, tokenizer=self.tok, device=self.device)
        self.arg_model = MaraArgumentModel(d_model=self.cfg.d_model).to(self.device)
        if os.path.exists(ARG_CKPT_PATH):
            self.arg_model.load_state_dict(torch.load(ARG_CKPT_PATH, map_location=self.device, weights_only=False))
        self.arg_model.eval()
        self.tool_names = ["set_lights", "set_thermostat", "control_device", "lock_door", "none"]

    def predict_single(self, sq: str) -> List[Dict[str, Any]]:
        sq_enc = self.tok.encode(sq)
        if not sq_enc:
            return []
        ids = torch.tensor([sq_enc], device=self.device)
        pos = torch.arange(len(sq_enc), device=self.device).unsqueeze(0)
        rec = {
            "state": f"User Request: {sq}\nAvailable Functions: {', '.join(self.tool_names)}",
            "questions": [
                {
                    "instr": "Which function should be triggered?",
                    "options": [f"tool: {t}" for t in self.tool_names],
                    "label": 0,
                    "qtype": "choice",
                }
            ]
        }
        with torch.no_grad():
            probs = self.model.probs(self.tok, rec, device=self.device)
            top_tool_idx = probs[0].argmax().item()
            tool_name = self.tool_names[top_tool_idx]
            conf = probs[0][top_tool_idx].item()

            if tool_name == "none" or conf < self.tau:
                return []

            h, _ = self.model(ids, position_ids=pos)
            h_seq = h[0]
            h_dec = h_seq[-1]
            tokens = self.tok.tokenize(sq)
            raw_args = self.arg_model.extract_arguments(
                tool_name=tool_name,
                h_query=h_seq,
                h_decide=h_dec,
                query_text=sq,
                query_tokens=tokens,
            )

            clean_args = {}
            if tool_name == "set_lights":
                clean_args = {"room": raw_args.get("room", "living room"), "brightness": raw_args.get("brightness", 100)}
            elif tool_name == "set_thermostat":
                clean_args = {"temperature_c": raw_args.get("temperature_c", 22.0)}
            elif tool_name == "control_device":
                clean_args = {"device": raw_args.get("device", "fan"), "action": raw_args.get("action", "on")}
            elif tool_name == "lock_door":
                clean_args = {"door": raw_args.get("door", "front door"), "locked": raw_args.get("locked", True)}

            return [{"name": tool_name, "arguments": clean_args}]

    def predict(self, query: str, use_planner: bool = False) -> List[Dict[str, Any]]:
        if not use_planner:
            return self.predict_single(query)

        plan = self.planner.plan_tools(query)
        sub_queries = plan.get("sub_queries", [query])
        all_calls = []
        for sq in sub_queries:
            sub_calls = self.predict_single(sq)
            all_calls.extend(sub_calls)
        return all_calls


class NeedleEvaluator:
    def __init__(self, weights: Optional[str] = None, tau: float = 0.50):
        self.weights = weights
        self.tau = tau
        self.planner = ToolPlanner(None, None)

    def predict_single(self, query: str) -> List[Dict[str, Any]]:
        kwargs = {"tools": FAIR_TOOLS}
        if self.weights and os.path.exists(self.weights):
            kwargs["weights"] = self.weights
        agent = needle.Needle(**kwargs)
        res = agent.complete(query)
        conf = getattr(agent, "confidence", None)
        if conf is None and isinstance(res, dict):
            conf = res.get("confidence")
        
        # Confidence threshold gating if confidence head is present
        if conf is not None and conf < self.tau:
            return []

        calls = []
        raw_calls = res.get("function_calls", []) if isinstance(res, dict) else (getattr(agent, "function_calls", []) or [])
        for c in raw_calls:
            if isinstance(c, dict):
                calls.append({"name": c.get("name"), "arguments": c.get("arguments", {})})
            else:
                calls.append({"name": c.name, "arguments": c.arguments})
        return calls

    def predict(self, query: str, use_planner: bool = False) -> List[Dict[str, Any]]:
        if not use_planner:
            return self.predict_single(query)

        plan = self.planner.plan_tools(query)
        sub_queries = plan.get("sub_queries", [query])
        all_calls = []
        for sq in sub_queries:
            sub_calls = self.predict_single(sq)
            all_calls.extend(sub_calls)
        return all_calls


def run_benchmark_on_suite(config_name: str, predict_fn, suite: List[Dict[str, Any]]) -> Dict[str, Any]:
    categories = ["clean", "paraphrase", "typo_noise", "compound", "hard_negative"]
    stats = {c: {"total": 0, "correct": 0, "latencies": []} for c in categories}
    overall_total = 0
    overall_correct = 0
    all_latencies = []
    failures = []

    for idx, item in enumerate(suite):
        cat = item["category"]
        q = item["query"]
        gt = item["ground_truth"]

        t0 = time.perf_counter()
        pred = predict_fn(q)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        is_match = check_ordered_exact_match(pred, gt)

        stats[cat]["total"] += 1
        overall_total += 1
        all_latencies.append(lat_ms)
        stats[cat]["latencies"].append(lat_ms)

        if is_match:
            stats[cat]["correct"] += 1
            overall_correct += 1
        else:
            failures.append({
                "category": cat,
                "query": q,
                "predicted": pred,
                "ground_truth": gt,
            })

        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx+1}/{len(suite)} queries...", flush=True)

    # Summary table
    suite_metrics = {}
    for cat in categories:
        k = stats[cat]["correct"]
        n = stats[cat]["total"]
        p, ci_l, ci_u = wilson_interval(k, n)
        lats = stats[cat]["latencies"]
        suite_metrics[cat] = {
            "correct": k,
            "total": n,
            "accuracy": round(p * 100.0, 2),
            "ci_95": (round(ci_l * 100.0, 2), round(ci_u * 100.0, 2)),
            "latency_p50_ms": round(sorted(lats)[len(lats) // 2], 2) if lats else 0.0,
        }

    # Hard Negative False Positive Rate
    hn_n = stats["hard_negative"]["total"]
    hn_correct = stats["hard_negative"]["correct"]
    hn_fp = hn_n - hn_correct
    fp_rate, fp_l, fp_u = wilson_interval(hn_fp, hn_n)

    # Overall OEM
    ov_p, ov_l, ov_u = wilson_interval(overall_correct, overall_total)

    all_latencies.sort()
    result = {
        "config": config_name,
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": round(ov_p * 100.0, 2),
            "ci_95": (round(ov_l * 100.0, 2), round(ov_u * 100.0, 2)),
            "latency_mean_ms": round(sum(all_latencies) / len(all_latencies), 2),
            "latency_p50_ms": round(all_latencies[len(all_latencies) // 2], 2),
            "latency_p95_ms": round(all_latencies[int(len(all_latencies) * 0.95)], 2),
        },
        "false_positive_rate": {
            "fp_count": hn_fp,
            "total": hn_n,
            "rate": round(fp_rate * 100.0, 2),
            "ci_95": (round(fp_l * 100.0, 2), round(fp_u * 100.0, 2)),
        },
        "per_category": suite_metrics,
        "sample_failures": failures[:10],
    }
    return result


def run_all_ablations(needle_specialist_cact: str = "checkpoints/needle3_specialist.cact") -> Dict[str, Any]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        suite = [json.loads(l) for l in f if l.strip()]

    print(f"================================================================================")
    print(f"FROZEN INDEPENDENT TEST SUITE: {len(suite)} EXAMPLES (50 x 5 CATEGORIES)")
    print(f"================================================================================")

    mara_eval = MaraEvaluator(tau=0.50)
    needle_base_eval = NeedleEvaluator(weights=None, tau=0.50)

    has_specialist = os.path.exists(os.path.join(ROOT, needle_specialist_cact))
    if has_specialist:
        needle_spec_eval = NeedleEvaluator(weights=needle_specialist_cact, tau=0.50)
    else:
        print(f"[!] Warning: Needle Specialist weights '{needle_specialist_cact}' not found.")
        needle_spec_eval = None

    configs = [
        ("Mara AFM Alone (No Planner)", lambda q: mara_eval.predict(q, use_planner=False)),
        ("Mara AFM + Planner", lambda q: mara_eval.predict(q, use_planner=True)),
        ("Needle 3 Base (Stateless, Fair Tools, No Planner)", lambda q: needle_base_eval.predict(q, use_planner=False)),
        ("Needle 3 Base + Planner", lambda q: needle_base_eval.predict(q, use_planner=True)),
    ]

    if needle_spec_eval:
        configs.extend([
            ("Needle 3 Specialist (LoRA, No Planner)", lambda q: needle_spec_eval.predict(q, use_planner=False)),
            ("Needle 3 Specialist + Planner", lambda q: needle_spec_eval.predict(q, use_planner=True)),
        ])

    all_results = {}
    for name, fn in configs:
        print(f"\n[*] Evaluating: {name}...", flush=True)
        res = run_benchmark_on_suite(name, fn, suite)
        all_results[name] = res
        ov = res["overall"]
        fp = res["false_positive_rate"]
        print(f"    OEM: {ov['accuracy']}% (95% CI: [{ov['ci_95'][0]}%, {ov['ci_95'][1]}%]) | FP Rate: {fp['rate']}% | Latency (p50): {ov['latency_p50_ms']} ms", flush=True)
        for cat, data in res["per_category"].items():
            print(f"      - {cat:<15}: {data['accuracy']:>5.1f}% ({data['correct']}/{data['total']}) [CI: {data['ci_95'][0]}% - {data['ci_95'][1]}%]", flush=True)

    out_path = os.path.join(ROOT, "data", "frozen_ablation_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Full benchmark results saved to: {out_path}", flush=True)
    return all_results


if __name__ == "__main__":
    cact = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/needle3_specialist.cact"
    run_all_ablations(cact)
