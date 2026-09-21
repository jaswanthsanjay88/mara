"""
Dataset Generator for AFM & Needle 3 Head-to-Head Benchmark.
Generates balanced, diverse, challenging test datasets containing:
1. Category A: Clean Direct Requests
2. Category B: Lexical Paraphrases & Slang (varied verbs, idioms)
3. Category C: Typos & Speech Recognition Noise (character swaps, omissions)
4. Category D: Compound Multi-Step Requests (conjunctions, sequential actions)
5. Category E: Hard Negatives & Near-Misses (informational, shopping, chitchat)
"""

import json
import os
import random
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hand-Crafted & High-Fidelity Linguistic Variations
# ---------------------------------------------------------------------------

CLEAN_SAMPLES = [
    # set_lights
    ("turn on the living room light", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 100}}]),
    ("turn off the kitchen lights", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}}]),
    ("dim the living room to 30", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 30}}]),
    ("set bedroom lights to 50 percent", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 50}}]),
    ("turn on the bathroom light", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 100}}]),
    ("turn off the bedroom light", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 0}}]),
    ("set kitchen lights to 80", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 80}}]),
    ("dim bathroom lights to 20", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 20}}]),

    # set_thermostat
    ("set thermostat to 21", [{"name": "set_thermostat", "arguments": {"temperature_c": 21.0}}]),
    ("set temperature to 22.5 degrees", [{"name": "set_thermostat", "arguments": {"temperature_c": 22.5}}]),
    ("set the thermostat to 19 degrees celsius", [{"name": "set_thermostat", "arguments": {"temperature_c": 19.0}}]),
    ("change temperature to 20", [{"name": "set_thermostat", "arguments": {"temperature_c": 20.0}}]),
    ("set thermostat to 23", [{"name": "set_thermostat", "arguments": {"temperature_c": 23.0}}]),
    ("adjust thermostat to 18.5 degrees", [{"name": "set_thermostat", "arguments": {"temperature_c": 18.5}}]),

    # control_device
    ("turn on the fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("turn off the fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("open the garage door", [{"name": "control_device", "arguments": {"device": "garage door", "action": "open"}}]),
    ("close the garage door", [{"name": "control_device", "arguments": {"device": "garage door", "action": "close"}}]),

    # lock_door
    ("lock the front door", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("unlock the front door", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
]

PARAPHRASE_SLANG_SAMPLES = [
    # set_lights
    ("kill the lights in the kitchen", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}}]),
    ("make it darker in the lounge, drop it to 15", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 15}}]),
    ("illuminate the master bedroom fully", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 100}}]),
    ("shut down all the lamps in the bathroom", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 0}}]),
    ("crank the kitchen brightness up to 90", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 90}}]),
    ("blackout the living room", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 0}}]),
    ("soften the bedroom lighting down to 25", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 25}}]),
    ("light up the bathroom", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 100}}]),

    # set_thermostat
    ("it is freezing in here, crank the heat to 24", [{"name": "set_thermostat", "arguments": {"temperature_c": 24.0}}]),
    ("chill the place down to 18 celsius", [{"name": "set_thermostat", "arguments": {"temperature_c": 18.0}}]),
    ("make the room a comfortable 21 degrees", [{"name": "set_thermostat", "arguments": {"temperature_c": 21.0}}]),
    ("warm up the house to 22.5", [{"name": "set_thermostat", "arguments": {"temperature_c": 22.5}}]),
    ("cool down the bedroom, put the temp at 19", [{"name": "set_thermostat", "arguments": {"temperature_c": 19.0}}]),
    ("drop climate control target to 20", [{"name": "set_thermostat", "arguments": {"temperature_c": 20.0}}]),

    # control_device
    ("start the ceiling fan spinning", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("cut power to the fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("raise the garage shutter", [{"name": "control_device", "arguments": {"device": "garage door", "action": "open"}}]),
    ("shut down the garage entrance", [{"name": "control_device", "arguments": {"device": "garage door", "action": "close"}}]),

    # lock_door
    ("bolt the main entrance", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("unlatch the front door for me", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("secure the front entryway", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("disarm the front deadbolt", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
]

TYPO_NOISY_SAMPLES = [
    # set_lights
    ("trun of kitche light", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}}]),
    ("dim livng rom to 30", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 30}}]),
    ("set bedrom lites to 45", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 45}}]),
    ("turn on bathrom light", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 100}}]),
    ("turnn off livng room", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 0}}]),
    ("dim the kitchn to 15 pct", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 15}}]),
    ("swich on bedroom ligts", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 100}}]),
    ("shut off the bathrom ligth", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 0}}]),

    # set_thermostat
    ("set themostat to 22", [{"name": "set_thermostat", "arguments": {"temperature_c": 22.0}}]),
    ("chagne temp to 21.5 degres", [{"name": "set_thermostat", "arguments": {"temperature_c": 21.5}}]),
    ("set temparature to 19", [{"name": "set_thermostat", "arguments": {"temperature_c": 19.0}}]),
    ("thermostt 20 celsus", [{"name": "set_thermostat", "arguments": {"temperature_c": 20.0}}]),
    ("set hvac to 23 deg", [{"name": "set_thermostat", "arguments": {"temperature_c": 23.0}}]),
    ("set tempreture 18.5", [{"name": "set_thermostat", "arguments": {"temperature_c": 18.5}}]),

    # control_device
    ("turn on teh fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("trun off ceiling fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("opn the grage door", [{"name": "control_device", "arguments": {"device": "garage door", "action": "open"}}]),
    ("clsoe the garage dor", [{"name": "control_device", "arguments": {"device": "garage door", "action": "close"}}]),

    # lock_door
    ("lokk front dor", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("unlok the frnt door", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("loc the front entrance", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("unlck frontdoor", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
]

COMPOUND_SAMPLES = [
    (
        "turn off kitchen lights and lock the front door",
        [
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
        ]
    ),
    (
        "turn on the fan and set thermostat to 21",
        [
            {"name": "control_device", "arguments": {"device": "fan", "action": "on"}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 21.0}},
        ]
    ),
    (
        "dim bedroom lights to 20 and lock the front door",
        [
            {"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 20}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
        ]
    ),
    (
        "close the garage door and turn off living room lights",
        [
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 0}},
        ]
    ),
    (
        "set thermostat to 22 and turn on the fan",
        [
            {"name": "set_thermostat", "arguments": {"temperature_c": 22.0}},
            {"name": "control_device", "arguments": {"device": "fan", "action": "on"}},
        ]
    ),
    (
        "turn on kitchen lights and open the garage door",
        [
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 100}},
            {"name": "control_device", "arguments": {"device": "garage door", "action": "open"}},
        ]
    ),
    (
        "unlock the front door and turn on living room lights",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": False}},
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 100}},
        ]
    ),
    (
        "turn off the fan and lock the front door",
        [
            {"name": "control_device", "arguments": {"device": "fan", "action": "off"}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
        ]
    ),
    (
        "dim living room to 40 and set thermostat to 20.5",
        [
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 40}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 20.5}},
        ]
    ),
    (
        "shut kitchen lights and turn off the fan",
        [
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}},
            {"name": "control_device", "arguments": {"device": "fan", "action": "off"}},
        ]
    ),
]

HARD_NEGATIVE_SAMPLES = [
    ("what is the capital of france", []),
    ("how does a thermostat work in modern homes", []),
    ("is the kitchen light led or halogen", []),
    ("who invented the electric ceiling fan", []),
    ("where can i buy a replacement front door lock", []),
    ("can you order some pizza with extra pepperoni", []),
    ("tell me a story about a haunted garage", []),
    ("what is the outdoor temperature in tokyo right now", []),
    ("can dogs see the light spectrum from indoor bulbs", []),
    ("explain the second law of thermodynamics", []),
    ("why does the fan make a squeaking noise", []),
    ("summarize the plot of the great gatsby", []),
    ("what is 45 times 12", []),
    ("who was the first president of the united states", []),
    ("is there an app to monitor energy consumption", []),
    ("how do electronic deadbolts prevent lock bumping", []),
    ("what is the difference between celsius and fahrenheit", []),
    ("write a poem about midnight breezes and fans", []),
    ("are smart lights compatible with 5ghz wifi", []),
    ("recommend five good sci-fi movies on netflix", []),
]


def generate_benchmark_suite():
    """Generates the official held-out benchmark suite (100 items)."""
    suite = []

    # Category A: Clean (20)
    for q, calls in CLEAN_SAMPLES:
        suite.append({"category": "clean", "query": q, "ground_truth": calls})

    # Category B: Paraphrases & Slang (20)
    for q, calls in PARAPHRASE_SLANG_SAMPLES:
        suite.append({"category": "paraphrase", "query": q, "ground_truth": calls})

    # Category C: Typos & Noise (20)
    for q, calls in TYPO_NOISY_SAMPLES:
        suite.append({"category": "typo_noise", "query": q, "ground_truth": calls})

    # Category D: Compound Multi-Step (20 - 10 duplicated with minor variation)
    for q, calls in COMPOUND_SAMPLES:
        suite.append({"category": "compound", "query": q, "ground_truth": calls})
    for q, calls in COMPOUND_SAMPLES:
        suite.append({"category": "compound", "query": f"please {q}", "ground_truth": calls})

    # Category E: Hard Negatives (20)
    for q, calls in HARD_NEGATIVE_SAMPLES:
        suite.append({"category": "hard_negative", "query": q, "ground_truth": calls})

    out_path = os.path.join(DATA_DIR, "heldout_benchmark.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in suite:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(suite)} held-out benchmark samples to {out_path}")
    print("Breakdown:")
    cats = {}
    for item in suite:
        cats[item["category"]] = cats.get(item["category"], 0) + 1
    for k, v in cats.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    generate_benchmark_suite()
