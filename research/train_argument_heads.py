"""
Training script for Mara's Neural Argument Filling Heads (SpanPointerHead + EnumHead).
Trains:
1. Room Enum Head (cross-entropy)
2. Device Enum Head (cross-entropy)
3. Action Enum Head (cross-entropy)
4. Locked Status Head (cross-entropy)
5. Span Pointer Head (start & end token positions cross-entropy for numbers and values)
"""

import json
import os
import random
import re
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mara.model import Mara, MaraConfig, branch_mask_batch
from mara.tokenizer import load_tokenizer
from mara.data import TOKENIZER_PATH
from mara.argument_heads import MaraArgumentModel, ROOMS, DEVICES, ACTIONS

CKPT_PATH = os.path.join(ROOT, "checkpoints", "mara_afm.pt")
ARG_CKPT_PATH = os.path.join(ROOT, "checkpoints", "mara_arg_heads.pt")


def find_token_span(tokens: list, target_str: str) -> tuple:
    """Finds start and end token indices corresponding to target substring."""
    target_clean = str(target_str).strip().lower()
    for i, t in enumerate(tokens):
        t_clean = t.replace("Ġ", "").lower()
        if target_clean in t_clean or t_clean in target_clean:
            return i, i
    # Fallback to middle
    return len(tokens) // 2, len(tokens) // 2


def build_training_batch():
    """Builds synthetic supervised samples for slot filling."""
    samples = []

    # set_lights
    for r_idx, r in enumerate(ROOMS):
        for b in [0, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90, 100]:
            if b == 0:
                queries = [
                    f"turn off the {r} lights",
                    f"shut down lights in {r}",
                    f"kill the {r} lamps",
                    f"blackout the {r}",
                    f"extinguish {r} light",
                ]
            elif b == 100:
                queries = [
                    f"turn on {r} lights",
                    f"illuminate the {r}",
                    f"switch on {r} lamps",
                    f"brighten {r} to maximum",
                    f"light up the {r}",
                ]
            else:
                queries = [
                    f"dim the {r} to {b}",
                    f"set {r} brightness to {b} percent",
                    f"make {r} {b} percent",
                    f"adjust {r} lights to {b}",
                    f"drop {r} down to {b}",
                ]
            for q in queries:
                samples.append({
                    "tool": "set_lights",
                    "query": q,
                    "room_idx": r_idx,
                    "target_num": str(b),
                    "slot_id": 0,
                })

    # set_thermostat
    for t in [18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 24.0]:
        t_str = str(t)
        queries = [
            f"set thermostat to {t_str}",
            f"adjust climate target to {t_str} degrees",
            f"set temperature to {t_str} celsius",
            f"cool down to {t_str}",
            f"warm up the house to {t_str}",
            f"change temp to {t_str}",
        ]
        for q in queries:
            samples.append({
                "tool": "set_thermostat",
                "query": q,
                "target_num": t_str,
                "slot_id": 1,
            })

    # control_device
    for d_idx, d in enumerate(DEVICES):
        for a_idx, a in enumerate(ACTIONS):
            queries = [
                f"turn {a} the {d}",
                f"{a} {d}",
                f"please {a} the {d}",
                f"switch {d} to {a}",
            ]
            for q in queries:
                samples.append({
                    "tool": "control_device",
                    "query": q,
                    "device_idx": d_idx,
                    "action_idx": a_idx,
                })

    # lock_door
    for is_locked, l_idx in [(True, 0), (False, 1)]:
        if is_locked:
            queries = [
                "lock the front door",
                "bolt the front door",
                "secure the front entrance",
                "lock front door",
                "lock up the front",
            ]
        else:
            queries = [
                "unlock the front door",
                "unlatch the front door",
                "disarm the front deadbolt",
                "unlock front door",
                "open the front lock",
            ]
        for q in queries:
            samples.append({
                "tool": "lock_door",
                "query": q,
                "lock_idx": l_idx,
            })

    return samples


def train_argument_heads():
    device = torch.device("cpu")
    print(f"Training Mara Argument Heads on: {device}")

    # Load base model & tokenizer
    tok = load_tokenizer(TOKENIZER_PATH)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = MaraConfig(**ckpt["config"])
    base_model = Mara(cfg).to(device)
    base_model.load_state_dict(ckpt["model"])
    base_model.eval()

    # Create argument filling model
    arg_model = MaraArgumentModel(d_model=cfg.d_model).to(device)
    optimizer = torch.optim.AdamW(arg_model.parameters(), lr=1e-3, weight_decay=1e-4)

    samples = build_training_batch()
    print(f"Generated {len(samples)} training samples for argument extraction.")

    epochs = 12
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        random.shuffle(samples)

        for item in samples:
            q = item["query"]
            tool_name = item["tool"]

            # Tokenize query
            q_ids = tok.encode(q)
            tokens = tok.tokenize(q)

            if not q_ids:
                continue

            ids_tensor = torch.tensor([q_ids], device=device)
            pos_tensor = torch.arange(len(q_ids), device=device).unsqueeze(0)

            # Extract base representations
            with torch.no_grad():
                h, _ = base_model(ids_tensor, position_ids=pos_tensor)
                h_seq = h[0]
                h_decide = h_seq[-1]

            loss = torch.tensor(0.0, device=device)

            if tool_name == "set_lights":
                # Room loss
                room_logits = arg_model.room_head(h_decide)
                tgt_room = torch.tensor(item["room_idx"], device=device)
                loss = loss + F.cross_entropy(room_logits.unsqueeze(0), tgt_room.unsqueeze(0))

                # Span loss if number is in query
                if "target_num" in item:
                    s_idx, e_idx = find_token_span(tokens, item["target_num"])
                    slot_vec = arg_model.slot_embed(torch.tensor(item["slot_id"], device=device))
                    start_l, end_l = arg_model.span_head(h_seq, slot_vec)
                    loss = loss + F.cross_entropy(start_l.unsqueeze(0), torch.tensor([s_idx], device=device))
                    loss = loss + F.cross_entropy(end_l.unsqueeze(0), torch.tensor([e_idx], device=device))

            elif tool_name == "set_thermostat":
                s_idx, e_idx = find_token_span(tokens, item["target_num"])
                slot_vec = arg_model.slot_embed(torch.tensor(item["slot_id"], device=device))
                start_l, end_l = arg_model.span_head(h_seq, slot_vec)
                loss = loss + F.cross_entropy(start_l.unsqueeze(0), torch.tensor([s_idx], device=device))
                loss = loss + F.cross_entropy(end_l.unsqueeze(0), torch.tensor([e_idx], device=device))

            elif tool_name == "control_device":
                dev_logits = arg_model.device_head(h_decide)
                act_logits = arg_model.action_head(h_decide)
                tgt_dev = torch.tensor(item["device_idx"], device=device)
                tgt_act = torch.tensor(item["action_idx"], device=device)
                loss = loss + F.cross_entropy(dev_logits.unsqueeze(0), tgt_dev.unsqueeze(0))
                loss = loss + F.cross_entropy(act_logits.unsqueeze(0), tgt_act.unsqueeze(0))

            elif tool_name == "lock_door":
                lock_logits = arg_model.locked_head(h_decide)
                tgt_lock = torch.tensor(item["lock_idx"], device=device)
                loss = loss + F.cross_entropy(lock_logits.unsqueeze(0), tgt_lock.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(samples)
        if ep % 3 == 0 or ep == epochs:
            print(f"Epoch {ep}/{epochs} - Slot Loss: {avg_loss:.4f}")

    # Save trained argument heads
    torch.save(arg_model.state_dict(), ARG_CKPT_PATH)
    print(f"Saved trained Mara argument heads to {ARG_CKPT_PATH}")


if __name__ == "__main__":
    train_argument_heads()
