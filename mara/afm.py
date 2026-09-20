"""
Automation Foundation Model (AFM) Tool Calling Engine.
Provides:
1. @tool decorator for automated JSON schema generation from Python functions.
2. ToolRegistry for tool compilation, dispatch, and execution.
3. Embedded IoT & Microcontroller hardware primitives (GPIO, PWM, I2C, Sensors).
4. Dual-mode dispatch: Fast-path (TypeSafe pointer slots) & Flexible-path (JSON tokens).
"""

import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

# Type mapping from Python to JSON Schema
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class Tool:
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = (description or func.__doc__ or "").strip()
        self.schema = self._generate_schema()

    def _generate_schema(self) -> Dict[str, Any]:
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, str)
            json_type = TYPE_MAP.get(param_type, "string")
            properties[param_name] = {
                "type": json_type,
            }
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    """Registry managing available tools, prompt generation, and execution."""
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, func_or_tool: Union[Callable, Tool], name: Optional[str] = None, description: Optional[str] = None):
        if isinstance(func_or_tool, Tool):
            tool = func_or_tool
        else:
            tool = Tool(func_or_tool, name=name, description=description)
        self.tools[tool.name] = tool
        return tool

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [tool.schema for tool in self.tools.values()]

    def execute(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a single tool call dictionary: {'name': '...', 'arguments': {...}}."""
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})

        if name not in self.tools:
            return {"name": name, "error": f"Unknown tool '{name}'"}

        try:
            res = self.tools[name](**args)
            return {"name": name, "result": res}
        except Exception as e:
            return {"name": name, "error": str(e)}

    def compile_fast_path_record(self, user_query: str, target_tool: Optional[str] = None, target_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compiles the query and tools into a TypeSafe Decision record for Sub-5ms bilinear evaluation.
        Evaluates tool routing + parameter slots in a single pass.
        """
        tool_names = list(self.tools.keys()) + ["none"]
        state = f"User Request: {user_query}\nAvailable Functions: {', '.join(tool_names)}"

        questions = [
            {
                "instr": "Which function should be triggered to fulfill this automation request?",
                "options": [f"tool: {t}" for t in tool_names],
                "label": tool_names.index(target_tool) if (target_tool and target_tool in tool_names) else (len(tool_names) - 1),
                "qtype": "choice",
            }
        ]

        if target_tool and target_tool in self.tools and target_args:
            schema = self.tools[target_tool].schema["parameters"]["properties"]
            for param_name, prop in schema.items():
                val = target_args.get(param_name)
                # Parameter binding slot
                questions.append({
                    "instr": f"Parameter value for '{param_name}'",
                    "options": [f"value: {val}" if val is not None else "null"],
                    "label": 0,
                    "qtype": "choice",
                })

        return {"state": state, "questions": questions}


# Global default registry
default_registry = ToolRegistry()


def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to register functions as AFM tools."""
    def decorator(fn: Callable) -> Tool:
        return default_registry.register(fn, name=name, description=description)
    return decorator


# ---------------------------------------------------------------------------
# Embedded Hardware, Microcontroller & Smart-Home Primitives
# ---------------------------------------------------------------------------

_MOCK_GPIO: Dict[int, int] = {}
_MOCK_PWM: Dict[int, Dict[str, int]] = {}
_MOCK_DEVICES: Dict[str, Dict[str, Any]] = {
    "kitchen_light": {"room": "kitchen", "device": "light", "state": "off", "brightness": 0},
    "living_room_ac": {"room": "living room", "device": "ac", "state": "off", "temperature": 24},
    "front_door_lock": {"room": "entrance", "device": "lock", "state": "locked"},
}


@tool(name="gpio_write", description="Sets digital output pin state on microcontroller (0=LOW, 1=HIGH)")
def gpio_write(pin: int, value: int) -> str:
    val = 1 if int(value) > 0 else 0
    _MOCK_GPIO[pin] = val
    return f"GPIO {pin} set to {'HIGH' if val else 'LOW'}"


@tool(name="gpio_read", description="Reads digital input pin state on microcontroller")
def gpio_read(pin: int) -> dict:
    val = _MOCK_GPIO.get(pin, 0)
    return {"pin": pin, "value": val, "state": "HIGH" if val else "LOW"}


@tool(name="pwm_set", description="Configures PWM output duty cycle (0-100%) and frequency on specified pin")
def pwm_set(pin: int, duty_cycle: int, frequency: int = 1000) -> str:
    duty = max(0, min(100, int(duty_cycle)))
    _MOCK_PWM[pin] = {"duty_cycle": duty, "frequency": frequency}
    return f"PWM on pin {pin}: duty={duty}%, freq={frequency}Hz"


@tool(name="control_device", description="Controls smart home devices (light, fan, ac, lock)")
def control_device(room: str, device: str, action: str, level: int = 100) -> dict:
    key = f"{room.lower().replace(' ', '_')}_{device.lower()}"
    if key not in _MOCK_DEVICES:
        _MOCK_DEVICES[key] = {"room": room, "device": device, "state": "off"}

    d = _MOCK_DEVICES[key]
    act = action.lower()
    if "on" in act:
        d["state"] = "on"
        d["level"] = level
    elif "off" in act:
        d["state"] = "off"
    elif "dim" in act or "set" in act:
        d["state"] = "on"
        d["level"] = level
    return {"device": key, "status": d}


@tool(name="read_sensor", description="Queries environmental telemetry sensors (temperature, humidity, motion)")
def read_sensor(sensor_type: str, location: str) -> dict:
    return {
        "sensor": sensor_type,
        "location": location,
        "temperature_c": 22.5,
        "humidity_percent": 45,
        "status": "normal",
    }


@tool(name="schedule_timer", description="Schedules an automation command after specified duration in seconds")
def schedule_timer(duration_seconds: int, command: str) -> str:
    return f"Scheduled '{command}' to execute in {duration_seconds} seconds"
