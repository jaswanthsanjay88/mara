"""
Unit tests for Automation Foundation Model (AFM) engine:
- @tool decorator & JSON Schema generation
- Tool execution & hardware mock validation
- AFM prompt formatting and tool call parsing
- Fast-path tool routing and dispatch
"""

import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mara.afm import (
    ToolRegistry,
    tool,
    default_registry,
    gpio_write,
    gpio_read,
    pwm_set,
    control_device,
    read_sensor,
)
from mara.data import TOKENIZER_PATH
from mara.tokenizer import (
    format_afm_prompt,
    parse_afm_tool_calls,
    load_tokenizer,
)
from mara.model import Mara, MaraConfig


def test_tool_decorator_and_schema():
    custom_reg = ToolRegistry()

    @custom_reg.register
    def set_relay(relay_id: int, state: bool, pulse_ms: int = 100) -> str:
        """Toggles an electrical relay switch."""
        return f"Relay {relay_id} set to {state}"

    schemas = custom_reg.get_schemas()
    assert len(schemas) == 1
    s = schemas[0]
    assert s["name"] == "set_relay"
    assert s["description"] == "Toggles an electrical relay switch."
    assert s["parameters"]["type"] == "object"
    props = s["parameters"]["properties"]
    assert props["relay_id"]["type"] == "integer"
    assert props["state"]["type"] == "boolean"
    assert props["pulse_ms"]["type"] == "integer"
    assert "relay_id" in s["parameters"]["required"]
    assert "state" in s["parameters"]["required"]
    assert "pulse_ms" not in s["parameters"]["required"]  # Has default value

    # Test execution
    res = custom_reg.execute({"name": "set_relay", "arguments": {"relay_id": 3, "state": True}})
    assert res["name"] == "set_relay"
    assert res["result"] == "Relay 3 set to True"


def test_hardware_mock_primitives():
    # 1. GPIO write and read
    res_w = default_registry.execute({"name": "gpio_write", "arguments": {"pin": 14, "value": 1}})
    assert res_w["result"] == "GPIO 14 set to HIGH"

    res_r = default_registry.execute({"name": "gpio_read", "arguments": {"pin": 14}})
    assert res_r["result"]["value"] == 1
    assert res_r["result"]["state"] == "HIGH"

    # 2. PWM
    res_pwm = default_registry.execute({"name": "pwm_set", "arguments": {"pin": 18, "duty_cycle": 75, "frequency": 2000}})
    assert "duty=75%" in res_pwm["result"]

    # 3. Smart home control
    res_dev = default_registry.execute({"name": "control_device", "arguments": {"room": "kitchen", "device": "light", "action": "dim", "level": 40}})
    assert res_dev["result"]["status"]["level"] == 40
    assert res_dev["result"]["status"]["state"] == "on"

    # 4. Unknown tool error handling
    res_err = default_registry.execute({"name": "non_existent_tool", "arguments": {}})
    assert "error" in res_err


def test_afm_prompt_formatting_and_parsing():
    schemas = [{"name": "gpio_write", "description": "Write GPIO", "parameters": {}}]
    query = "Set GPIO 12 to HIGH"
    calls = [{"name": "gpio_write", "arguments": {"pin": 12, "value": 1}}]

    prompt = format_afm_prompt(schemas, query, tool_calls=calls)
    assert "<|tools|>" in prompt
    assert "</tools>" in prompt
    assert "<|call|>" in prompt
    assert "</call>" in prompt

    parsed = parse_afm_tool_calls(prompt)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "gpio_write"
    assert parsed[0]["arguments"]["pin"] == 12


def test_fast_path_model_routing():
    device = torch.device("cpu")
    tok = load_tokenizer(TOKENIZER_PATH)
    cfg = MaraConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2)
    model = Mara(cfg).to(device)
    model.eval()

    # Route and execute
    query = "Set pin 15 to HIGH"
    result = model.route_and_execute_tool(tok, query, default_registry, device=device)
    assert "status" in result
    assert "tool" in result
    assert "confidence" in result
    assert result["confidence"] >= 0.0


if __name__ == "__main__":
    test_tool_decorator_and_schema()
    print("test_tool_decorator_and_schema PASSED")
    test_hardware_mock_primitives()
    print("test_hardware_mock_primitives PASSED")
    test_afm_prompt_formatting_and_parsing()
    print("test_afm_prompt_formatting_and_parsing PASSED")
    test_fast_path_model_routing()
    print("test_fast_path_model_routing PASSED")
