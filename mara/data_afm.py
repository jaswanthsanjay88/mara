"""
Automation Foundation Model (AFM) Dataset Generator.
Generates multi-domain tool-calling traces and decision records covering:
- Hardware microcontroller controls (GPIO, PWM, ADC)
- Smart-home ambient IoT orchestration
- Multi-tool parallel invocations
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

ROOMS = ["kitchen", "master bedroom", "living room", "bathroom", "garage", "study", "porch", "hallway"]
DEVICES = ["light", "fan", "ac", "heater", "lock", "blinds"]
SENSOR_TYPES = ["temperature", "humidity", "motion", "co2", "lux"]

CHITCHAT_QUERIES = [
    "What is the capital of France?",
    "Tell me a short poem about sunrise.",
    "Explain what Ohm's law is in simple terms.",
    "Who painted the Mona Lisa?",
    "How does a brushless DC motor work?",
    "Hello, how are you doing today?",
    "Can you help me solve 15 multiplied by 8?",
]


def gen_afm_traces(n: int = 5000, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates AFM training traces in both:
    1. Full sequence format (<|tools|> ... <|call|>)
    2. DecisionMara pointer records for fast-path evaluation
    """
    rng = random.Random(seed)
    traces = []
    decision_records = []
    schemas = default_registry.get_schemas()
    tool_names = list(default_registry.tools.keys())

    for i in range(n):
        trace_type = rng.choices(["gpio", "pwm", "device", "sensor", "timer", "chitchat", "parallel"],
                                 weights=[20, 15, 30, 15, 10, 5, 5])[0]

        if trace_type == "gpio":
            pin = rng.choice([2, 4, 5, 12, 13, 14, 15, 18, 19, 21, 22, 23, 25, 26, 27])
            val = rng.choice([0, 1])
            is_write = rng.choice([True, True, False])
            if is_write:
                query = rng.choice([
                    f"Set GPIO pin {pin} to {'HIGH' if val else 'LOW'}.",
                    f"Write {'1' if val else '0'} to digital pin {pin}.",
                    f"Turn {'on' if val else 'off'} output pin {pin}.",
                ])
                call = {"name": "gpio_write", "arguments": {"pin": pin, "value": val}}
                target_tool = "gpio_write"
                target_args = {"pin": pin, "value": val}
            else:
                query = f"Read the current logic level of GPIO pin {pin}."
                call = {"name": "gpio_read", "arguments": {"pin": pin}}
                target_tool = "gpio_read"
                target_args = {"pin": pin}

            calls = [call]

        elif trace_type == "pwm":
            pin = rng.choice([16, 17, 18, 19, 21, 22])
            duty = rng.randint(0, 100)
            freq = rng.choice([500, 1000, 5000, 10000])
            query = rng.choice([
                f"Set PWM duty cycle on pin {pin} to {duty} percent.",
                f"Drive motor PWM pin {pin} at {duty}% with frequency {freq} Hz.",
            ])
            calls = [{"name": "pwm_set", "arguments": {"pin": pin, "duty_cycle": duty, "frequency": freq}}]
            target_tool = "pwm_set"
            target_args = {"pin": pin, "duty_cycle": duty, "frequency": freq}

        elif trace_type == "device":
            room = rng.choice(ROOMS)
            device = rng.choice(DEVICES)
            action = rng.choice(["turn_on", "turn_off", "dim", "set_temp"])
            if action == "turn_on":
                query = f"Turn on the {device} in the {room}."
                args = {"room": room, "device": device, "action": "turn_on", "level": 100}
            elif action == "turn_off":
                query = f"Switch off the {room} {device}."
                args = {"room": room, "device": device, "action": "turn_off", "level": 0}
            elif action == "dim":
                level = rng.randint(10, 90)
                query = f"Dim the {room} {device} to {level}%."
                args = {"room": room, "device": device, "action": "dim", "level": level}
            else:
                temp = rng.randint(18, 28)
                query = f"Set the {room} AC to {temp} degrees."
                args = {"room": room, "device": "ac", "action": "set_temp", "level": temp}

            calls = [{"name": "control_device", "arguments": args}]
            target_tool = "control_device"
            target_args = args

        elif trace_type == "sensor":
            sensor = rng.choice(SENSOR_TYPES)
            room = rng.choice(ROOMS)
            query = f"Check the {sensor} sensor in the {room}."
            calls = [{"name": "read_sensor", "arguments": {"sensor_type": sensor, "location": room}}]
            target_tool = "read_sensor"
            target_args = {"sensor_type": sensor, "location": room}

        elif trace_type == "timer":
            secs = rng.choice([10, 30, 60, 120, 300, 600])
            cmd = f"turn off living room light"
            query = f"In {secs // 60 if secs >= 60 else secs} {'minutes' if secs >= 60 else 'seconds'}, {cmd}."
            calls = [{"name": "schedule_timer", "arguments": {"duration_seconds": secs, "command": cmd}}]
            target_tool = "schedule_timer"
            target_args = {"duration_seconds": secs, "command": cmd}

        elif trace_type == "parallel":
            # Compound command
            room = rng.choice(ROOMS)
            pin = rng.choice([12, 14, 27])
            query = f"Turn off the {room} light and set pin {pin} to HIGH."
            calls = [
                {"name": "control_device", "arguments": {"room": room, "device": "light", "action": "turn_off", "level": 0}},
                {"name": "gpio_write", "arguments": {"pin": pin, "value": 1}},
            ]
            target_tool = "control_device"
            target_args = {"room": room, "device": "light"}

        else:
            # Chitchat / Negative example (No tool should be called)
            query = rng.choice(CHITCHAT_QUERIES)
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


def build_and_save_afm_dataset(num_samples: int = 5000):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Generating {num_samples:,} AFM training traces...")
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
