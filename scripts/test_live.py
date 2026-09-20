import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer, encode_record
from mara.afm import default_registry

device = torch.device("cpu")
tok = load_tokenizer("data/tokenizer.json")
ckpt = torch.load("checkpoints/mara_afm.pt", map_location=device, weights_only=False)
cfg = MaraConfig(**ckpt["config"])
model = Mara(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

queries = [
    "Turn off the kitchen light.",
    "Set GPIO pin 14 to HIGH.",
    "Check the temperature sensor in the living room.",
    "Drive motor PWM pin 18 at 75% duty cycle.",
    "Set the living room AC to 22 degrees.",
    "Lock the front door lock.",
    "What is the capital of France?",
    "Tell me a short poem."
]

tool_names = list(default_registry.tools.keys()) + ["none"]
print("Testing Real Mara AFM Model Weights:")
print("-" * 55)

for q in queries:
    rec = default_registry.compile_fast_path_record(q)
    packed = encode_record(tok, rec)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs, _ = model.forward_decision(packed, device=device)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    p = probs[0]
    best_idx = p.argmax().item()
    top_tool = tool_names[best_idx]
    confidence = p[best_idx].item() * 100
    print(f"[{elapsed_ms:5.2f}ms] {top_tool:<15} ({confidence:5.1f}%): \"{q}\"")
