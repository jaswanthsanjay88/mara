"""
Fine-tunes Mara on the canonical smart home tool suite:
['set_lights', 'set_thermostat', 'control_device', 'lock_door', 'none']
Jointly trains:
1. Tool PointerHead (which tool should be called)
2. Room Enum Head (which room for set_lights)
3. Device Enum Head (fan vs garage door for control_device)
4. Action Enum Head (on, off, open, close)
5. Lock Enum Head (locked vs unlocked)
6. Span Pointer Head (brightness and temperature numeric values)
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
FINE_CKPT_PATH = os.path.join(ROOT, "checkpoints", "mara_smart_home.pt")
ARG_CKPT_PATH = os.path.join(ROOT, "checkpoints", "mara_arg_heads.pt")

TOOL_NAMES = ["set_lights", "set_thermostat", "control_device", "lock_door", "none"]


def find_token_span(tokens: list, target_str: str) -> tuple:
    target_clean = str(target_str).strip().lower()
    clean_tokens = [t.replace("Ġ", " ").strip().lower() for t in tokens]
    # 1. Exact single token match
    for i, t in enumerate(clean_tokens):
        if t == target_clean:
            return i, i
    # 2. Exact multi-token span match
    for span_len in range(1, min(6, len(tokens) + 1)):
        for i in range(len(tokens) - span_len + 1):
            sub_text = "".join(tokens[i : i + span_len]).replace("Ġ", "").strip().lower()
            if target_clean == sub_text:
                return i, i + span_len - 1
    # 3. Substring contained in multi-token span
    for span_len in range(1, min(6, len(tokens) + 1)):
        for i in range(len(tokens) - span_len + 1):
            sub_text = "".join(tokens[i : i + span_len]).replace("Ġ", "").strip().lower()
            if target_clean in sub_text:
                return i, i + span_len - 1
    # 4. Token substring fallback
    for i, t in enumerate(clean_tokens):
        if t and (t in target_clean or target_clean in t):
            return i, i
    return len(tokens) // 2, len(tokens) // 2


def build_finetune_dataset(n_samples=1500):
    """Generates training queries with supervised tool and slot labels."""
    samples = []
    rng = random.Random(42)

    ROOM_VARIATIONS = {
        0: ["living room", "lounge", "main room", "sitting room", "livng room"],
        1: ["kitchen", "galley", "kitchin", "kitchn", "kitche"],
        2: ["bedroom", "master bedroom", "bed room", "bedrom"],
        3: ["bathroom", "washroom", "restroom", "bathrom"],
        4: ["garage", "garag"],
    }

    # 1. set_lights (450 samples)
    verbs_on = ["turn on", "switch on", "light up", "brighten", "illuminate", "enable", "activate", "swich on", "trun on"]
    verbs_off = ["turn off", "switch off", "shut down", "kill", "blackout", "extinguish", "darken", "shut off", "cut power to", "trun off", "turnn off", "trun of"]
    verbs_dim = ["dim", "set", "adjust", "drop", "lower", "change", "soften", "crank"]

    for _ in range(450):
        r_idx = rng.randint(0, len(ROOMS) - 1)
        room_name = rng.choice(ROOM_VARIATIONS[r_idx])
        mode = rng.choice(["on", "off", "dim"])

        if mode == "off":
            v = rng.choice(verbs_off)
            q = f"{v} the {room_name} lights" if rng.random() < 0.7 else f"{v} {room_name}"
            samples.append({
                "query": q,
                "tool_idx": 0,
                "room_idx": r_idx,
                "target_num": "0",
                "is_on": False,
            })
        elif mode == "on":
            v = rng.choice(verbs_on)
            q = f"{v} the {room_name} lights" if rng.random() < 0.7 else f"{v} {room_name}"
            samples.append({
                "query": q,
                "tool_idx": 0,
                "room_idx": r_idx,
                "target_num": "100",
                "is_on": True,
            })
        else:
            v = rng.choice(verbs_dim)
            b = rng.choice([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 85, 90])
            suffix = "%" if rng.random() < 0.25 else (" percent" if rng.random() < 0.25 else (" pct" if rng.random() < 0.25 else ""))
            q = f"{v} the {room_name} to {b}{suffix}" if rng.random() < 0.5 else f"{v} {room_name} lights to {b}{suffix}"
            samples.append({
                "query": q,
                "tool_idx": 0,
                "room_idx": r_idx,
                "target_num": str(b),
                "is_on": True,
            })

    # 2. set_thermostat (350 samples)
    t_verbs = [
        "set thermostat to", "adjust temperature to", "change temp to", "set climate to",
        "cool down to", "warm up to", "chill to", "crank heat to", "make it",
        "drop climate control to", "set themostat to", "chagne temp to", "set temparature to",
        "thermostt", "set hvac to", "set tempreture", "adjust thermostat to"
    ]
    for _ in range(350):
        v = rng.choice(t_verbs)
        t = rng.choice([18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0])
        unit = " degrees" if rng.random() < 0.25 else (" celsius" if rng.random() < 0.25 else (" deg" if rng.random() < 0.2 else ""))
        # Sometimes integer string if .0
        t_str = str(int(t)) if (t.is_integer() and rng.random() < 0.5) else str(t)
        q = f"{v} {t_str}{unit}"
        samples.append({
            "query": q,
            "tool_idx": 1,
            "target_num": t_str,
        })

    # 3. control_device (350 samples)
    fan_synonyms = ["fan", "ceiling fan", "exhaust fan", "cooler fan", "teh fan"]
    garage_synonyms = ["garage door", "garage shutter", "garage gate", "garage entrance", "grage door", "garage dor"]
    for _ in range(350):
        dev_idx = rng.choice([0, 1])  # 0: fan, 1: garage door
        if dev_idx == 0:
            dev_name = rng.choice(fan_synonyms)
            act_idx = rng.choice([0, 1])  # on or off
            act_str = "on" if act_idx == 0 else "off"
            v = rng.choice(["turn", "switch", "start", "spin"]) if act_idx == 0 else rng.choice(["turn", "switch", "cut power to", "stop", "trun"])
            if v in ["start", "spin"]:
                q = f"{v} the {dev_name}"
            elif v == "cut power to":
                q = f"cut power to the {dev_name}"
            else:
                q = f"{v} {act_str} the {dev_name}"
        else:
            dev_name = rng.choice(garage_synonyms)
            act_idx = rng.choice([2, 3])  # 2: open, 3: close
            if act_idx == 2:
                v = rng.choice(["open", "raise", "lift", "roll up", "opn"])
            else:
                v = rng.choice(["close", "shut", "shut down", "lower", "roll down", "clsoe"])
            q = f"{v} the {dev_name}"

        samples.append({
            "query": q,
            "tool_idx": 2,
            "device_idx": dev_idx,
            "action_idx": act_idx,
        })

    # 4. lock_door (250 samples)
    lock_synonyms = ["lock", "bolt", "secure", "latch", "lock up", "lokk", "loc"]
    unlock_synonyms = ["unlock", "unlatch", "disarm", "open lock", "unlok", "unlck"]
    door_synonyms = ["front door", "front entrance", "main entrance", "front entryway", "main door", "frnt door", "frontdoor"]
    for _ in range(250):
        is_locked = rng.choice([True, False])
        door = rng.choice(door_synonyms)
        if is_locked:
            v = rng.choice(lock_synonyms)
            q = f"{v} the {door}" if rng.random() < 0.7 else f"{v} {door}"
            l_idx = 0
        else:
            v = rng.choice(unlock_synonyms)
            q = f"{v} the {door}" if rng.random() < 0.7 else f"{v} {door}"
            l_idx = 1

        samples.append({
            "query": q,
            "tool_idx": 3,
            "lock_idx": l_idx,
        })

    # 5. none / Hard Negatives & Near Misses (250 samples)
    neg_templates = [
        "what is the capital of {country}",
        "tell me a story about {noun}",
        "how does a {device} work in modern homes",
        "who invented the electric {device}",
        "where can i buy a replacement {device}",
        "can you order some pizza with {topping}",
        "what is the outdoor temperature in {city} right now",
        "how far is earth from {planet}",
        "explain {concept} in simple words",
        "is the {device} energy efficient",
        "why does the {device} make a strange noise",
        "are smart {device} units safe from cyber attacks",
        "what is 45 times 12",
        "who was the first president of the united states",
        "recommend five good sci-fi movies on netflix",
        "summarize the plot of hamlet",
        "can dogs see the light spectrum from indoor bulbs",
    ]
    for _ in range(250):
        tpl = rng.choice(neg_templates)
        q = tpl.format(
            country=rng.choice(["france", "spain", "japan", "brazil", "canada", "germany", "italy"]),
            noun=rng.choice(["dragons", "space rockets", "forests", "pirates", "haunted house"]),
            device=rng.choice(["thermostat", "ceiling fan", "light bulb", "lock", "front door lock"]),
            topping=rng.choice(["cheese", "mushrooms", "pepperoni", "olives", "pineapple"]),
            city=rng.choice(["tokyo", "paris", "london", "sydney", "berlin"]),
            planet=rng.choice(["mars", "jupiter", "venus", "saturn"]),
            concept=rng.choice(["quantum physics", "photosynthesis", "thermodynamics", "gravity"]),
        )
        samples.append({
            "query": q,
            "tool_idx": 4,  # none
        })

    return samples


def finetune_mara():
    device = torch.device("cpu")
    print(f"Fine-tuning Mara on 4-tool schema on: {device}")

    tok = load_tokenizer(TOKENIZER_PATH)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = MaraConfig(**ckpt["config"])
    model = Mara(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()

    arg_model = MaraArgumentModel(d_model=cfg.d_model).to(device)
    if os.path.exists(ARG_CKPT_PATH):
        arg_model.load_state_dict(torch.load(ARG_CKPT_PATH, map_location=device, weights_only=False))
    arg_model.train()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arg_model.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )

    samples = build_finetune_dataset(n_samples=1200)
    print(f"Generated {len(samples)} training samples across 5 classes.")

    epochs = 6
    for ep in range(1, epochs + 1):
        random.shuffle(samples)
        total_loss = 0.0

        for item in samples:
            q = item["query"]
            target_tool_idx = item["tool_idx"]

            # 1. Forward decision record for tool routing
            rec = {
                "state": f"User Request: {q}\nAvailable Functions: {', '.join(TOOL_NAMES)}",
                "questions": [
                    {
                        "instr": "Which function should be triggered?",
                        "options": [f"tool: {t}" for t in TOOL_NAMES],
                        "label": target_tool_idx,
                        "qtype": "choice",
                    }
                ]
            }

            from mara.tokenizer import encode_record
            packed = encode_record(tok, rec)
            _, tool_loss = model.forward_decision(packed, device=device)

            loss = tool_loss

            # 2. Argument loss if not 'none'
            if target_tool_idx < 4:
                q_ids = tok.encode(q)
                tokens = tok.tokenize(q)
                if q_ids:
                    ids_tensor = torch.tensor([q_ids], device=device)
                    pos_tensor = torch.arange(len(q_ids), device=device).unsqueeze(0)
                    h, _ = model(ids_tensor, position_ids=pos_tensor)
                    h_seq = h[0]
                    h_dec = h_seq[-1]

                    if target_tool_idx == 0:  # set_lights
                        room_logits = arg_model.room_head(h_dec)
                        tgt_room = torch.tensor(item["room_idx"], device=device)
                        loss = loss + 0.5 * F.cross_entropy(room_logits.unsqueeze(0), tgt_room.unsqueeze(0))

                        if "target_num" in item:
                            s_idx, e_idx = find_token_span(tokens, item["target_num"])
                            slot_vec = arg_model.slot_embed(torch.tensor(0, device=device))
                            start_l, end_l = arg_model.span_head(h_seq, slot_vec)
                            loss = loss + 0.5 * (
                                F.cross_entropy(start_l.unsqueeze(0), torch.tensor([s_idx], device=device))
                                + F.cross_entropy(end_l.unsqueeze(0), torch.tensor([e_idx], device=device))
                            )

                    elif target_tool_idx == 1:  # set_thermostat
                        s_idx, e_idx = find_token_span(tokens, item["target_num"])
                        slot_vec = arg_model.slot_embed(torch.tensor(1, device=device))
                        start_l, end_l = arg_model.span_head(h_seq, slot_vec)
                        loss = loss + 0.5 * (
                            F.cross_entropy(start_l.unsqueeze(0), torch.tensor([s_idx], device=device))
                            + F.cross_entropy(end_l.unsqueeze(0), torch.tensor([e_idx], device=device))
                        )

                    elif target_tool_idx == 2:  # control_device
                        dev_logits = arg_model.device_head(h_dec)
                        act_logits = arg_model.action_head(h_dec)
                        loss = loss + 0.5 * (
                            F.cross_entropy(dev_logits.unsqueeze(0), torch.tensor(item["device_idx"], device=device).unsqueeze(0))
                            + F.cross_entropy(act_logits.unsqueeze(0), torch.tensor(item["action_idx"], device=device).unsqueeze(0))
                        )

                    elif target_tool_idx == 3:  # lock_door
                        lock_logits = arg_model.locked_head(h_dec)
                        loss = loss + 0.5 * F.cross_entropy(lock_logits.unsqueeze(0), torch.tensor(item["lock_idx"], device=device).unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(samples)
        print(f"Epoch {ep}/{epochs} - Joint Loss: {avg_loss:.4f}")

    # Save fine-tuned checkpoints
    ckpt["model"] = model.state_dict()
    torch.save(ckpt, FINE_CKPT_PATH)
    torch.save(arg_model.state_dict(), ARG_CKPT_PATH)
    print(f"Successfully saved fine-tuned weights to {FINE_CKPT_PATH} and {ARG_CKPT_PATH}")


if __name__ == "__main__":
    finetune_mara()
