"""
Mara AFM: Multi-Step Tool Planning & Compound Orchestration Example.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer
from mara.data import TOKENIZER_PATH
from mara.planner import ToolPlanner


def main():
    device = torch.device("cpu")

    # 1. Load model & tokenizer
    tok = load_tokenizer(TOKENIZER_PATH)
    ckpt = torch.load("checkpoints/mara_afm.pt", map_location=device, weights_only=False)
    cfg = MaraConfig(**ckpt["config"])
    model = Mara(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    planner = ToolPlanner(model=model, tokenizer=tok, device=device)

    # 2. Test a multi-action compound query
    compound_query = "turn off living room lights and lock the front door"
    print("==================================================")
    print(f"Goal Request: '{compound_query}'")
    print("==================================================")

    plan = planner.plan_tools(compound_query)
    print(f"Plan Strategy : {plan['strategy']}")
    print(f"Steps Planned : {len(plan['sub_queries'])}")
    for idx, (sq, reason) in enumerate(zip(plan["sub_queries"], plan["reasons"]), start=1):
        print(f"  Step {idx}: {sq} -> ({reason})")

    # 3. Test a macro intent
    macro_query = "good night"
    print("\n==================================================")
    print(f"Macro Request: '{macro_query}'")
    print("==================================================")

    macro_plan = planner.plan_tools(macro_query)
    print(f"Goal Name     : {macro_plan['goal']}")
    print(f"Description   : {macro_plan['description']}")
    print(f"Steps Planned : {len(macro_plan['sub_queries'])}")
    for idx, (sq, reason) in enumerate(zip(macro_plan["sub_queries"], macro_plan["reasons"]), start=1):
        print(f"  Step {idx}: {sq} -> {reason}")


if __name__ == "__main__":
    main()
