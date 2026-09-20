"""
Data generation, preparation, and streaming for DecisionMara.
Combines:
1. Multi-domain decision records (noul, choice, score)
2. Smart-home device registry & control records
3. High-signal educational reasoning & domain text
"""

import json
import os
import random
from typing import Any, Iterator
import numpy as np
from tqdm import tqdm

from .tokenizer import (
    train_tokenizer,
    load_tokenizer,
    encode_record,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
RECORDS_JSONL = os.path.join(DATA_DIR, "decision_records.jsonl")

ROOMS = ["kitchen", "master bedroom", "living room", "bathroom", "garage", "study", "balcony", "basement"]
DEVICES = ["light", "fan", "ac", "heater", "speaker", "tv", "lock", "blinds"]
OWNERS = ["Divya", "Arjun", "Meera", "Ravi", "Anita", "Kiran", "Sarah", "David"]


def gen_smart_home_records(n: int = 5000, seed: int = 42) -> list[dict[str, Any]]:
    """Generates synthetic smart-home state inspection and tool control decision records."""
    rng = random.Random(seed)
    records = []

    for _ in range(n):
        room = rng.choice(ROOMS)
        device = rng.choice(DEVICES)
        owner = rng.choice(OWNERS)
        state = rng.choice(["on", "off", "standby", "error"])
        level = rng.randint(10, 100)
        temp = rng.randint(16, 30)

        doc = (
            f"Device Registry Report:\n"
            f"Location: {room}\n"
            f"Device: {device}\n"
            f"Registered Owner: {owner}\n"
            f"Operational Status: {state}\n"
            f"Current Level: {level}%\n"
            f"Ambient Temperature: {temp}°C\n"
        )

        # Question 1: Choice (Device type)
        opt_devs = rng.sample(DEVICES, 4)
        if device not in opt_devs:
            opt_devs[0] = device
        rng.shuffle(opt_devs)
        q1 = {
            "instr": "Which device is described in this registry entry?",
            "options": [f"device: {d}" for d in opt_devs],
            "label": opt_devs.index(device),
            "qtype": "choice",
        }

        # Question 2: Noul (Boolean operational check)
        check_active = (state == "on")
        q2 = {
            "instr": f"Is the {device} currently active and powered on?",
            "options": ["false: device is off or standby", "true: device is active"],
            "label": 1 if check_active else 0,
            "qtype": "noul",
        }

        # Question 3: Score (Power level rating 1-5)
        bucket = min(4, level // 20)
        q3 = {
            "instr": f"Rate the operating capacity of the {device}",
            "options": ["1: minimal (<20%)", "2: low (20-40%)", "3: medium (40-60%)", "4: high (60-80%)", "5: maximum (80-100%)"],
            "label": bucket,
            "qtype": "score",
        }

        records.append({
            "state": doc,
            "questions": [q1, q2, q3],
        })

    return records


def gen_enterprise_decision_records(n: int = 5000, seed: int = 1337) -> list[dict[str, Any]]:
    """Generates customer support, policy, and business triage decision records."""
    rng = random.Random(seed)
    records = []

    departments = ["billing", "technical_support", "sales_inquiry", "trust_and_safety"]
    urgencies = ["low", "medium", "urgent"]

    scenarios = [
        ("I was charged $120 twice on my credit card for invoice #9821. Please reverse immediately.", "billing", 2, "frustrated"),
        ("The API gateway is throwing 502 Bad Gateway errors for all European endpoints.", "technical_support", 2, "neutral"),
        ("We are a team of 400 engineers evaluating enterprise plans. We need pricing for custom SSO.", "sales_inquiry", 1, "calm"),
        ("Someone posted offensive and threatening harassment messages in our public channel.", "trust_and_safety", 2, "furious"),
        ("How do I update my profile picture in the desktop app settings?", "technical_support", 0, "calm"),
        ("Can we get a copy of your SOC2 Type II compliance certificate?", "sales_inquiry", 0, "neutral"),
        ("My annual renewal processed today but I requested cancellation last week.", "billing", 1, "frustrated"),
    ]

    for _ in range(n):
        text, dept, urg_level, mood = rng.choice(scenarios)
        # Add random ticket metadata
        ticket_id = rng.randint(10000, 99999)
        doc = f"Support Ticket #{ticket_id}:\nCustomer Message: \"{text}\"\nUser Tier: {rng.choice(['Free', 'Pro', 'Enterprise'])}"

        q_dept = {
            "instr": "Route this ticket to the appropriate operational team",
            "options": [f"department: {d}" for d in departments],
            "label": departments.index(dept),
            "qtype": "choice",
        }
        q_urgent = {
            "instr": "Is this ticket considered critical/urgent priority?",
            "options": ["false: standard turnaround", "true: critical SLA required"],
            "label": 1 if urg_level >= 2 else 0,
            "qtype": "noul",
        }
        q_score = {
            "instr": "Assess customer escalation severity level",
            "options": ["Level 1: Routine inquiry", "Level 2: Moderate concern", "Level 3: Critical executive escalation"],
            "label": min(2, urg_level),
            "qtype": "score",
        }

        records.append({
            "state": doc,
            "questions": [q_dept, q_urgent, q_score],
        })

    return records


def prepare_dataset(
    num_samples: int = 15_000,
    vocab_size: int = VOCAB_SIZE,
    sample_docs_for_tok: int = 5_000,
):
    """
    1. Generates rich multi-domain decision records.
    2. Trains a custom BPE tokenizer with reserved decision delimiters.
    3. Packs records into binary uint16 memmap files (train.bin, val.bin).
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating multi-domain decision dataset...")
    data_home = gen_smart_home_records(n=num_samples // 2)
    data_biz = gen_enterprise_decision_records(n=num_samples // 2)
    all_records = data_home + data_biz
    random.seed(42)
    random.shuffle(all_records)

    with open(RECORDS_JSONL, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(all_records):,} records to {RECORDS_JSONL}")

    # Train tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Training byte-level BPE tokenizer (vocab {vocab_size})...")
        sample_texts = []
        for r in all_records[:sample_docs_for_tok]:
            sample_texts.append(r["state"])
            for q in r["questions"]:
                sample_texts.append(q["instr"])
                sample_texts.extend(q["options"])
        tok = train_tokenizer(sample_texts, TOKENIZER_PATH, vocab_size=vocab_size)
        print(f"Tokenizer trained. Vocab: {tok.vocab_size}")
    else:
        tok = load_tokenizer(TOKENIZER_PATH)
        print(f"Loaded existing tokenizer from {TOKENIZER_PATH}")

    # Split train/val
    split_idx = int(len(all_records) * 0.95)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]

    def write_binary_packed(records: list[dict[str, Any]], out_path: str):
        ids_buffer = []
        with open(out_path, "wb") as f:
            for rec in tqdm(records, desc=f"packing {os.path.basename(out_path)}"):
                enc = encode_record(tok, rec)
                # End with EOS (0)
                ids_buffer.extend(enc["ids"] + [0])
                while len(ids_buffer) >= 500_000:
                    np.asarray(ids_buffer[:500_000], dtype=np.uint16).tofile(f)
                    ids_buffer = ids_buffer[500_000:]
            if ids_buffer:
                np.asarray(ids_buffer, dtype=np.uint16).tofile(f)

    write_binary_packed(train_records, TRAIN_BIN)
    write_binary_packed(val_records, VAL_BIN)

    print(f"Dataset complete!")
    print(f"Train tokens: {os.path.getsize(TRAIN_BIN) // 2:,}")
    print(f"Val tokens:   {os.path.getsize(VAL_BIN) // 2:,}")


if __name__ == "__main__":
    prepare_dataset()
