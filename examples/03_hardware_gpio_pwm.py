"""
Mara AFM: Hardware Primitives (GPIO, PWM, Telemetry) Dispatch Example.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mara.afm import default_registry
from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer
from mara.data import TOKENIZER_PATH


def main():
    device = torch.device("cpu")

    # 1. Load model & tokenizer
    tok = load_tokenizer(TOKENIZER_PATH)
    ckpt = torch.load("checkpoints/mara_afm.pt", map_location=device, weights_only=False)
    cfg = MaraConfig(**ckpt["config"])
    model = Mara(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 2. Hardware test queries
    queries = [
        "set gpio pin 14 to high",
        "set pwm on pin 5 to 75 percent duty cycle",
        "read ambient temperature sensor on bus 1",
    ]

    print("==================================================")
    print("      Mara AFM Hardware Primitive Dispatch       ")
    print("==================================================")

    tool_names = list(default_registry.tools.keys()) + ["none"]

    for q in queries:
        record = default_registry.compile_fast_path_record(q)
        with torch.no_grad():
            probs = model.probs(tok, record, device=device)

        pred_idx = probs[0].argmax().item()
        conf = probs[0][pred_idx].item()
        tool_name = tool_names[pred_idx]

        print(f"Query     : '{q}'")
        print(f"Target    : {tool_name}")
        print(f"Confidence: {conf * 100:.1f}%\n")


if __name__ == "__main__":
    main()
