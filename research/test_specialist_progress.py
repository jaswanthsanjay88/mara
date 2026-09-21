import json
import os
import sys
import time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import needle
from research.reextract_stateless_fair_failures import FAIR_TOOLS
DATA_PATH = os.path.join(ROOT, "data", "frozen_eval_250.jsonl")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    suite = [json.loads(l) for l in f if l.strip()]

print(f"Testing test_needle.cact on {len(suite)} queries with flush...", flush=True)

# Test first 30 queries
for idx, s in enumerate(suite[:30]):
    q = s["query"]
    t0 = time.time()
    print(f"[{idx+1}/30] Running: '{q}'", end="", flush=True)
    agent = needle.Needle(weights="checkpoints/test_needle.cact", tools=FAIR_TOOLS)
    res = agent.complete(q)
    dt = time.time() - t0
    calls = res.get("function_calls", [])
    print(f" -> {calls} ({dt:.2f}s)", flush=True)

print("Batch of 30 complete!", flush=True)
