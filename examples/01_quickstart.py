"""
Mara AFM Quickstart: Minimal On-Device Tool Calling Example.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mara.afm import tool, default_registry
from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer
from mara.data import TOKENIZER_PATH


# 1. Define tools using standard Python typing and docstrings
@tool
def set_lights(room: str, brightness: int = 100) -> str:
    """Adjust brightness for a specific room light."""
    return f"Set {room} brightness to {brightness}%"


@tool
def lock_door(door: str, locked: bool = True) -> str:
    """Lock or unlock an exterior security door."""
    status = "locked" if locked else "unlocked"
    return f"Door {door} is now {status}"


def main():
    device = torch.device("cpu")

    # 2. Load model & tokenizer
    tok = load_tokenizer(TOKENIZER_PATH)
    ckpt = torch.load("checkpoints/mara_afm.pt", map_location=device, weights_only=False)
    cfg = MaraConfig(**ckpt["config"])
    model = Mara(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Mara AFM loaded successfully ({model.num_params():,} parameters).")

    # 3. User request
    query = "dim the living room light to 30 percent"
    print(f"\nUser Query: '{query}'")

    # 4. Compile fast-path decision record and infer routing probabilities
    record = default_registry.compile_fast_path_record(query)
    with torch.no_grad():
        probs = model.probs(tok, record, device=device)

    tool_names = list(default_registry.tools.keys()) + ["none"]
    pred_idx = probs[0].argmax().item()
    confidence = probs[0][pred_idx].item()
    selected_tool = tool_names[pred_idx]

    print(f"Selected Tool: {selected_tool} (Confidence: {confidence * 100:.1f}%)")

    # 5. Dispatch execution
    if selected_tool == "set_lights":
        result = set_lights(room="living room", brightness=30)
        print(f"Execution Result: {result}")


if __name__ == "__main__":
    main()
