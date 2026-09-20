import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer, encode_record
from mara.afm import default_registry
from mara.planner import ToolPlanner

device = torch.device("cpu")
tok = load_tokenizer("data/tokenizer.json")
ckpt = torch.load("checkpoints/mara_afm.pt", map_location=device, weights_only=False)
model = Mara(MaraConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

planner = ToolPlanner(model, tok, device)

test_queries = [
    "turn on kitchen lights",
    "turn off kitchen lights and lock front door",
    "good night",
    "turn on living lights and set gpio pin 14 to HIGH"
]

print("Testing AFM Tool Planning Engine:")
print("=" * 60)

for q in test_queries:
    plan_spec = planner.plan_tools(q)
    print(f"\nUser Query: \"{q}\"")
    print(f"Plan Goal : {plan_spec['goal']} ({len(plan_spec['sub_queries'])} steps)")

    for step_idx, sub_q in enumerate(plan_spec["sub_queries"], start=1):
        rec = default_registry.compile_fast_path_record(sub_q)
        packed = encode_record(tok, rec)
        t0 = time.perf_counter()
        with torch.no_grad():
            probs, _ = model.forward_decision(packed, device=device)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        tool_names = list(default_registry.tools.keys()) + ["none"]
        best_idx = probs[0].argmax().item()
        chosen = tool_names[best_idx]
        conf = probs[0][best_idx].item() * 100
        print(f"  Step {step_idx}: [{chosen:<14}] {sub_q:<42} ({conf:5.1f}% | {elapsed_ms:4.2f}ms)")
