import argparse
import json
import os
import random

ROOMS = ["kitchen", "bedroom", "living", "bathroom", "garage", "study", "hallway", "balcony"]
DEVICES = ["light", "fan", "ac", "heater", "speaker", "tv"]
OWNERS = ["Divya", "Arjun", "Meera", "Ravi", "Anita", "Kiran"]


def build_fact_world(seed: int = 42, n_devices: int = 24):
    rng = random.Random(seed)
    facts, qa = [], []
    devices = {}
    for i in range(n_devices):
        room = rng.choice(ROOMS)
        kind = rng.choice(DEVICES)
        code = f"{kind[0].upper()}{kind[1:]}-{i:02d}"
        owner = rng.choice(OWNERS)
        state = rng.choice(["on", "off"])
        brightness = rng.randint(10, 100)
        devices[code] = {"room": room, "kind": kind, "owner": owner, "state": state, "brightness": brightness}
        facts.append(f"The {code} is a {kind} located in the {room}, owned by {owner}. It is currently {state} at {brightness} percent.")
    qa_templates = [
        ("Which room is the {code} in?", "room", ROOMS),
        ("Who owns the {code}?", "owner", OWNERS),
        ("What kind of device is the {code}?", "kind", DEVICES),
    ]
    for code, attrs in devices.items():
        q_t, attr_key, options = rng.choice(qa_templates)
        correct_val = attrs[attr_key]
        qa.append({
            "code": code,
            "question": q_t.format(code=code),
            "answer": correct_val,
            "options": options,
            "label": options.index(correct_val),
        })
    return facts, qa, devices


def build_fact_decision_records(seed: int = 42, n_devices: int = 24) -> list[dict]:
    """Generates evaluation records formatted for DecisionMara."""
    facts, qa, _ = build_fact_world(seed=seed, n_devices=n_devices)
    state = "\n".join(facts)
    questions = []
    for item in qa:
        questions.append({
            "instr": item["question"],
            "options": [f"{opt}" for opt in item["options"]],
            "label": item["label"],
            "qtype": "choice",
        })
    return [{"state": state, "questions": questions}]


def write_facts_corpus(facts, out_path, tokenizer=None, repeat: int = 40):
    text = "\n\n".join(facts) + "\n\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text * repeat)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    facts, qa, _ = build_fact_world()
    print(f"facts: {len(facts)}, qa: {len(qa)}")
    print("sample:", facts[0])
    print("qa:", qa[0])
