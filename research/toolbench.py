import argparse
import itertools
import json
import os
import random

ACTIONS = {
    "light": ["turn_on", "turn_off", "set_brightness"],
    "fan": ["turn_on", "turn_off", "set_speed"],
    "ac": ["turn_on", "turn_off", "set_temperature"],
}
ROOMS = ["kitchen", "bedroom", "living room", "bathroom", "garage", "study"]
COMMAND_TEMPLATES = [
    ("{action_phrase} the {room} {device}", True),
    ("in the {room}, {action_phrase} the {device}", True),
    ("{device} in the {room} — {action_phrase} it", True),
]

def gen(n: int = 2000, seed: int = 7):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        device = rng.choice(list(ACTIONS))
        action = rng.choice(ACTIONS[device])
        room = rng.choice(ROOMS)
        args = {"room": room}
        if action == "set_brightness":
            value = rng.randint(5, 100)
            phrase = f"dim to {value} percent"
            args["brightness"] = value
        elif action == "set_speed":
            value = rng.choice([1, 2, 3])
            phrase = f"set speed to {value}"
            args["speed"] = value
        elif action == "set_temperature":
            value = rng.randint(16, 30)
            phrase = f"cool to {value} degrees"
            args["temperature"] = value
        else:
            phrase = "switch on" if action == "turn_on" else "switch off"
        template = rng.choice(COMMAND_TEMPLATES)[0]
        command = template.format(action_phrase=phrase, room=room, device=device)
        out.append({
            "query": command,
            "tools": [{"name": "control_device",
                       "parameters": {"type": "object",
                                      "properties": {"room": {"type": "string"},
                                                     "brightness": {"type": "integer"},
                                                     "speed": {"type": "integer"},
                                                     "temperature": {"type": "integer"}}}}],
            "answers": [{"name": "control_device", "arguments": args}],
        })
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="data/toolbench.jsonl")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = gen(args.n)
    with open(args.out, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    print(f"wrote {len(data)} examples -> {args.out}")
