"""
Automation Foundation Model (AFM) Dataset Generator.
Generates multi-domain tool-calling traces and decision records covering:
- Hardware microcontroller controls (GPIO, PWM, ADC)
- Smart-home ambient IoT orchestration (lights, thermostat, locks, blinds)
- Natural human phrasing diversity (lower/upper/title case, with/without punctuation,
  plural/singular nouns, colloquial abbreviations)
- Negative / no-action chitchat controls
"""

import json
import os
import random
from typing import Any, Dict, List, Tuple

from .afm import default_registry
from .tokenizer import format_afm_prompt

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
AFM_JSONL = os.path.join(DATA_DIR, "afm_traces.jsonl")
AFM_DECISIONS_JSONL = os.path.join(DATA_DIR, "afm_decisions.jsonl")

ROOMS = ["kitchen", "master bedroom", "living room", "bathroom", "garage", "study", "porch", "hallway", "bedroom"]
DEVICES_SINGULAR = {
    "light": "light",
    "fan": "fan",
    "ac": "ac",
    "heater": "heater",
    "lock": "lock",
    "blinds": "blinds",
}
DEVICES_PLURAL = {
    "light": "lights",
    "fan": "fans",
    "ac": "ac units",
    "heater": "heaters",
    "lock": "locks",
    "blinds": "blinds",
}
SENSOR_TYPES = ["temperature", "humidity", "motion", "co2", "lux"]

CHITCHAT_QUERIES = [
    "what is the capital of france",
    "tell me a short poem about sunrise",
    "explain what ohm's law is in simple terms",
    "who painted the mona lisa",
    "how does a brushless dc motor work",
    "hello how are you doing today",
    "can you help me solve 15 multiplied by 8",
    "what is the speed of light in a vacuum",
    "recommend three science fiction novels",
    "how do plants convert sunlight into energy",
    "who discovered penicillin",
    "what is the difference between ram and rom",
    "write a quick haiku about autumn",
    "what year was the python programming language released",
    "explain quantum entanglement simply",
    "how many planets are in the solar system",
    "can dogs eat blueberries",
    "what is the deepest part of the ocean",
    "who was the first person on the moon",
    "explain how a transformer neural network works",
    "what is the boiling point of water in fahrenheit",
    "give me a recipe for chocolate chip cookies",
    "how does gps locate a phone",
    "why is the ocean salty",
    "summarize the plot of hamlet in two sentences",
    "what is the distance between earth and mars",
    "good morning assistant",
    "thank you for your help",
    "what is 42 plus 58",
    "how do airplanes fly",
    "who wrote pride and prejudice",
    "what causes earthquakes",
    "explain recursion in programming",
    "what is an api in web development",
    "can you tell me a funny joke",
]


def apply_case_and_punct(text: str, rng: random.Random, is_question: bool = False) -> str:
    """Applies realistic human chat variations in casing and punctuation."""
    c = rng.random()
    if c < 0.45:
        # lowercase
        t = text.lower()
    elif c < 0.85:
        # Capitalized first letter
        t = text[0].upper() + text[1:]
    elif c < 0.95:
        # Title Case
        t = text.title()
    else:
        # ALL CAPS
        t = text.upper()

    p = rng.random()
    if p < 0.50:
        # No ending punctuation (typical chat)
        return t
    elif p < 0.90:
        return t + ("?" if is_question else ".")
    else:
        return t + "!"


def gen_afm_traces(n: int = 5000, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates AFM training traces covering natural human phrasing variations
    across microcontroller primitives and smart-home ambient automations.
    """
    rng = random.Random(seed)
    traces = []
    decision_records = []
    schemas = default_registry.get_schemas()

    for i in range(n):
        trace_type = rng.choices(
            ["gpio", "pwm", "device", "sensor", "timer", "chitchat", "parallel"],
            weights=[20, 12, 38, 12, 8, 7, 3]
        )[0]

        if trace_type == "gpio":
            pin = rng.choice([2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27])
            val = rng.choice([0, 1])
            val_str = "HIGH" if val else "LOW"
            is_write = rng.choice([True, True, False])

            if is_write:
                templates = [
                    f"set gpio pin {pin} to {val_str}",
                    f"set gpio {pin} to {val_str}",
                    f"set pin {pin} to {val_str}",
                    f"set pin {pin} {val_str}",
                    f"write {val} to digital pin {pin}",
                    f"write {val} to gpio {pin}",
                    f"turn {'on' if val else 'off'} output pin {pin}",
                    f"turn {'on' if val else 'off'} pin {pin}",
                    f"gpio {pin} {val_str}",
                    f"pull pin {pin} {val_str}",
                    f"digital write pin {pin} {val}",
                ]
                call = {"name": "gpio_write", "arguments": {"pin": pin, "value": val}}
                target_tool = "gpio_write"
                target_args = {"pin": pin, "value": val}
            else:
                templates = [
                    f"read the current logic level of gpio pin {pin}",
                    f"read gpio pin {pin}",
                    f"read pin {pin}",
                    f"read gpio {pin}",
                    f"get state of pin {pin}",
                    f"check pin {pin} logic level",
                    f"gpio {pin} status",
                    f"is pin {pin} high or low",
                ]
                call = {"name": "gpio_read", "arguments": {"pin": pin}}
                target_tool = "gpio_read"
                target_args = {"pin": pin}

            raw_query = rng.choice(templates)
            query = apply_case_and_punct(raw_query, rng)
            calls = [call]

        elif trace_type == "pwm":
            pin = rng.choice([16, 17, 18, 19, 21, 22])
            duty = rng.randint(0, 100)
            freq = rng.choice([500, 1000, 5000, 10000])
            templates = [
                f"set pwm duty cycle on pin {pin} to {duty} percent",
                f"set pwm on pin {pin} to {duty}%",
                f"drive motor pwm pin {pin} at {duty}% with frequency {freq} hz",
                f"pwm pin {pin} {duty}%",
                f"set duty cycle on pin {pin} to {duty}%",
                f"configure pwm on pin {pin} at {duty}% duty and {freq}hz",
            ]
            raw_query = rng.choice(templates)
            query = apply_case_and_punct(raw_query, rng)
            calls = [{"name": "pwm_set", "arguments": {"pin": pin, "duty_cycle": duty, "frequency": freq}}]
            target_tool = "pwm_set"
            target_args = {"pin": pin, "duty_cycle": duty, "frequency": freq}

        elif trace_type == "device":
            room = rng.choice(ROOMS)
            dev_key = rng.choice(["light", "fan", "ac", "blinds", "lock"])
            # 50% singular, 50% plural
            dev_noun = DEVICES_PLURAL[dev_key] if rng.random() < 0.5 else DEVICES_SINGULAR[dev_key]
            action = rng.choice(["turn_on", "turn_off", "dim", "climate", "security"])

            if action == "turn_on":
                templates = [
                    f"turn on {room} {dev_noun}",
                    f"turn on the {room} {dev_noun}",
                    f"turn {room} {dev_noun} on",
                    f"turn on the {dev_noun} in the {room}",
                    f"turn on {dev_noun} in {room}",
                    f"switch on {room} {dev_noun}",
                    f"switch on the {room} {dev_noun}",
                    f"switch {room} {dev_noun} on",
                    f"power on {room} {dev_noun}",
                    f"{room} {dev_noun} on",
                    f"activate {room} {dev_noun}",
                    f"enable {room} {dev_noun}",
                ]
                if dev_key == "light":
                    templates.extend([
                        f"lights on in {room}",
                        f"turn on all lights in the {room}",
                        f"illuminate the {room}",
                    ])
                raw_query = rng.choice(templates)
                args = {"room": room, "device": dev_key, "action": "turn_on", "level": 100}

            elif action == "turn_off":
                templates = [
                    f"turn off {room} {dev_noun}",
                    f"turn off the {room} {dev_noun}",
                    f"turn {room} {dev_noun} off",
                    f"turn off the {dev_noun} in the {room}",
                    f"turn off {dev_noun} in {room}",
                    f"switch off {room} {dev_noun}",
                    f"switch off the {room} {dev_noun}",
                    f"switch {room} {dev_noun} off",
                    f"power off {room} {dev_noun}",
                    f"{room} {dev_noun} off",
                    f"shut down {room} {dev_noun}",
                    f"kill {room} {dev_noun}",
                    f"disable {room} {dev_noun}",
                ]
                if dev_key == "light":
                    templates.extend([
                        f"lights off in {room}",
                        f"turn off all lights in the {room}",
                    ])
                raw_query = rng.choice(templates)
                args = {"room": room, "device": dev_key, "action": "turn_off", "level": 0}

            elif action == "dim":
                level = rng.randint(10, 90)
                templates = [
                    f"dim {room} {dev_noun} to {level}%",
                    f"dim the {room} {dev_noun} to {level}%",
                    f"dim {room} {dev_noun} to {level} percent",
                    f"dim the {dev_noun} in the {room} to {level}%",
                    f"set {room} {dev_noun} to {level}%",
                    f"set {room} {dev_noun} brightness to {level}%",
                    f"adjust {room} {dev_noun} to {level}%",
                    f"{room} {dev_noun} {level}%",
                ]
                raw_query = rng.choice(templates)
                args = {"room": room, "device": dev_key, "action": "dim", "level": level}

            elif action == "climate":
                temp = rng.randint(18, 28)
                templates = [
                    f"set the {room} ac to {temp} degrees",
                    f"set {room} ac to {temp}",
                    f"set thermostat to {temp} degrees",
                    f"set thermostat to {temp}",
                    f"set temperature to {temp} in the {room}",
                    f"make the {room} {temp} degrees",
                    f"adjust {room} climate to {temp}",
                    f"change thermostat to {temp}C",
                ]
                raw_query = rng.choice(templates)
                args = {"room": room, "device": "ac", "action": "set_temp", "level": temp}

            else:  # security / blinds
                sub = rng.choice(["lock", "unlock", "blinds_open", "blinds_close"])
                if sub == "lock":
                    door = rng.choice(["front door", "back door", "garage door", "front door lock"])
                    raw_query = rng.choice([
                        f"lock {door}",
                        f"lock the {door}",
                        f"secure {door}",
                    ])
                    args = {"room": "entrance", "device": "lock", "action": "lock"}
                elif sub == "unlock":
                    door = rng.choice(["front door", "back door", "garage door", "front door lock"])
                    raw_query = rng.choice([
                        f"unlock {door}",
                        f"unlock the {door}",
                        f"open the {door} lock",
                    ])
                    args = {"room": "entrance", "device": "lock", "action": "unlock"}
                elif sub == "blinds_open":
                    raw_query = rng.choice([
                        f"open {room} blinds",
                        f"open the {room} blinds",
                        f"open the blinds in {room}",
                    ])
                    args = {"room": room, "device": "blinds", "action": "open"}
                else:
                    raw_query = rng.choice([
                        f"close {room} blinds",
                        f"close the {room} blinds",
                        f"shut {room} blinds",
                        f"close the blinds in the {room}",
                    ])
                    args = {"room": room, "device": "blinds", "action": "close"}

            query = apply_case_and_punct(raw_query, rng)
            calls = [{"name": "control_device", "arguments": args}]
            target_tool = "control_device"
            target_args = args

        elif trace_type == "sensor":
            sensor = rng.choice(SENSOR_TYPES)
            room = rng.choice(ROOMS)
            templates = [
                f"check the {sensor} sensor in the {room}",
                f"check {sensor} in {room}",
                f"what is the {sensor} in the {room}",
                f"read {sensor} sensor in {room}",
                f"query {sensor} in {room}",
                f"get {sensor} reading for {room}",
            ]
            raw_query = rng.choice(templates)
            query = apply_case_and_punct(raw_query, rng, is_question=True)
            calls = [{"name": "read_sensor", "arguments": {"sensor_type": sensor, "location": room}}]
            target_tool = "read_sensor"
            target_args = {"sensor_type": sensor, "location": room}

        elif trace_type == "timer":
            secs = rng.choice([10, 30, 60, 120, 300, 600])
            unit = "minutes" if secs >= 60 else "seconds"
            amt = secs // 60 if secs >= 60 else secs
            cmd = f"turn off {rng.choice(ROOMS)} light"
            templates = [
                f"in {amt} {unit}, {cmd}",
                f"after {amt} {unit} {cmd}",
                f"schedule timer for {amt} {unit} to {cmd}",
                f"set a timer for {amt} {unit} then {cmd}",
            ]
            raw_query = rng.choice(templates)
            query = apply_case_and_punct(raw_query, rng)
            calls = [{"name": "schedule_timer", "arguments": {"duration_seconds": secs, "command": cmd}}]
            target_tool = "schedule_timer"
            target_args = {"duration_seconds": secs, "command": cmd}

        elif trace_type == "parallel":
            room = rng.choice(ROOMS)
            pin = rng.choice([12, 14, 27])
            raw_query = f"turn off {room} lights and set pin {pin} to HIGH"
            query = apply_case_and_punct(raw_query, rng)
            calls = [
                {"name": "control_device", "arguments": {"room": room, "device": "light", "action": "turn_off", "level": 0}},
                {"name": "gpio_write", "arguments": {"pin": pin, "value": 1}},
            ]
            target_tool = "control_device"
            target_args = {"room": room, "device": "light"}

        else:
            # Chitchat / Negative example (No tool should be called)
            base_q = rng.choice(CHITCHAT_QUERIES)
            query = apply_case_and_punct(base_q, rng, is_question=True)
            calls = []
            target_tool = "none"
            target_args = {}

        # 1. Full AFM prompt sequence
        formatted_prompt = format_afm_prompt(schemas, query, tool_calls=calls)
        traces.append({
            "query": query,
            "tool_calls": calls,
            "text": formatted_prompt,
        })

        # 2. Fast-path decision record
        rec = default_registry.compile_fast_path_record(query, target_tool, target_args)
        decision_records.append(rec)

    return traces, decision_records


def build_and_save_afm_dataset(num_samples: int = 6000):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Generating {num_samples:,} AFM training traces with full linguistic diversity...")
    traces, decisions = gen_afm_traces(n=num_samples)

    with open(AFM_JSONL, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(traces):,} AFM traces to {AFM_JSONL}")

    with open(AFM_DECISIONS_JSONL, "w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")
    print(f"Saved {len(decisions):,} AFM decision records to {AFM_DECISIONS_JSONL}")


if __name__ == "__main__":
    build_and_save_afm_dataset()
