"""
Evaluation & Benchmarking harness for DecisionMara:
- Single-pass decision accuracy on FactWorld device registry
- Prefix KV-cache acceleration benchmarking (Cold vs Cached ms)
- Toolbench intent triage evaluation
"""

import argparse
import json
import os
import time
from typing import Any
import torch

from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer, encode_state
from research.factworld import build_fact_decision_records
from research.toolbench import gen as gen_toolbench


def evaluate_decision_probe(model: Mara, tok, records: list[dict[str, Any]], device: torch.device):
    """Evaluates decision records in a single forward pass per record."""
    total_q, correct_q = 0, 0
    t0 = time.time()

    for rec in records:
        probs = model.probs(tok, rec, device=device)
        for k, q in enumerate(rec["questions"]):
            pred = probs[k].argmax().item()
            correct = (pred == q["label"])
            correct_q += correct
            total_q += 1

    dt = (time.time() - t0) * 1000
    acc = (correct_q / total_q) if total_q > 0 else 0.0
    return {
        "accuracy": acc,
        "total_questions": total_q,
        "correct": correct_q,
        "latency_ms": dt,
        "ms_per_question": dt / max(1, total_q),
    }


def benchmark_kv_caching(model: Mara, tok, rec: dict[str, Any], device: torch.device):
    """Benchmarks Cold Document Prefill vs Prefix KV-Cached question evaluation."""
    # Step 1: Cold evaluation
    t0 = time.time()
    cold_probs = model.probs(tok, rec, device=device)
    cold_ms = (time.time() - t0) * 1000

    # Step 2: Prefill state into cache
    encoded_state = encode_state(tok, rec["state"])
    t_prefill = time.time()
    pkv, s_len = model.prefill_state(encoded_state["ids"], device=device)
    prefill_ms = (time.time() - t_prefill) * 1000

    # Step 3: Cached question evaluation
    t_cached = time.time()
    cached_probs = model.probs_cached(tok, rec, pkv, s_len, device=device)
    cached_ms = (time.time() - t_cached) * 1000

    speedup = (cold_ms / cached_ms) if cached_ms > 0 else 1.0

    return {
        "cold_ms": cold_ms,
        "prefill_ms": prefill_ms,
        "cached_ms": cached_ms,
        "speedup": speedup,
        "match": all((c - k).abs().max().item() < 1e-4 for c, k in zip(cold_probs, cached_probs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/mara_decision_base.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.ckpt} on {device}")

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint {args.ckpt} not found. Train the model first via: python -m mara.train")
        return

    tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = MaraConfig(**ckpt["config"])
    model = Mara(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Model loaded: {model.num_params():,} parameters, {cfg.n_layers} layers.")

    # 1. FactWorld Probe
    records = build_fact_decision_records()
    print("\n--- Running FactWorld Decision Probe ---")
    results = evaluate_decision_probe(model, tok, records, device=device)
    print(f"Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total_questions']})")
    print(f"Latency:  {results['latency_ms']:.2f} ms ({results['ms_per_question']:.2f} ms/question)")

    # 2. KV Cache Benchmark
    print("\n--- Benchmarking Prefix KV-Caching ---")
    bench = benchmark_kv_caching(model, tok, records[0], device=device)
    print(f"Cold Full Evaluation:   {bench['cold_ms']:.2f} ms")
    print(f"State Prefill Once:     {bench['prefill_ms']:.2f} ms")
    print(f"Cached Branch Query:    {bench['cached_ms']:.2f} ms (Speedup: {bench['speedup']:.1f}x)")
    print(f"Numerical Equivalence:  {bench['match']}")


if __name__ == "__main__":
    main()
