"""
AFM Tool Planner: Multi-Step Automation Planning & Orchestration Engine.
Decomposes complex, multi-action, or macro-intent automation goals into an ordered
execution graph of tool calls, evaluates each step via Mara AFM neural weights,
and executes them sequentially or in parallel.
"""

import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mara.afm import default_registry
from mara.tokenizer import encode_record

# Pre-compiled high-level macro routines
MACRO_ROUTINES = {
    "bedtime": {
        "goal": "Bedtime Routine",
        "description": "Prepare home for sleep: turn off public areas, dim bedroom, lock doors, set cool temp, arm alarm",
        "steps": [
            {"query": "turn off living room lights", "reason": "Shut off living room illumination"},
            {"query": "turn off kitchen lights", "reason": "Shut off kitchen illumination"},
            {"query": "dim bedroom lights to 20%", "reason": "Set dim night lighting in master bedroom"},
            {"query": "lock front door", "reason": "Secure primary entrance"},
            {"query": "set thermostat to 20 degrees", "reason": "Optimize overnight sleeping temperature"},
            {"query": "close blinds in bedroom", "reason": "Ensure privacy and block morning sun"},
        ]
    },
    "leave_home": {
        "goal": "Away / Leaving Home Routine",
        "description": "Secure the entire home, turn off all devices, lock doors, and activate alarm",
        "steps": [
            {"query": "turn off all lights", "reason": "Conserve power across all rooms"},
            {"query": "lock front door", "reason": "Secure front entrance"},
            {"query": "close blinds in bedroom", "reason": "Close all blinds for security"},
            {"query": "set thermostat to 18 degrees", "reason": "Switch HVAC to eco mode"},
        ]
    },
    "movie_mode": {
        "goal": "Cinema / Movie Mode",
        "description": "Adjust ambiance for screen viewing in living room",
        "steps": [
            {"query": "dim living room light to 15%", "reason": "Dim lighting to reduce glare"},
            {"query": "turn off kitchen lights", "reason": "Darken adjacent open kitchen"},
            {"query": "close blinds in bedroom", "reason": "Close window blinds"},
            {"query": "set thermostat to 21 degrees", "reason": "Comfortable room climate"},
        ]
    },
    "morning": {
        "goal": "Good Morning Routine",
        "description": "Wake-up environment: open blinds, turn on main lights, adjust climate",
        "steps": [
            {"query": "open blinds in bedroom", "reason": "Let in natural morning light"},
            {"query": "turn on kitchen lights", "reason": "Prepare kitchen for breakfast"},
            {"query": "turn on bedroom lights", "reason": "Brighten bedroom"},
            {"query": "set thermostat to 23 degrees", "reason": "Warm up house to morning temperature"},
        ]
    },
    "emergency": {
        "goal": "Emergency Safety Protocol",
        "description": "Immediate life-safety: unlock all doors, turn on all lights, sound hardware buzzer",
        "steps": [
            {"query": "unlock front door", "reason": "Enable rapid emergency egress"},
            {"query": "turn on all lights", "reason": "Maximize visibility in all rooms"},
            {"query": "set gpio pin 12 to HIGH", "reason": "Activate hardware siren buzzer"},
        ]
    }
}


def detect_macro_intent(query: str) -> Optional[Dict[str, Any]]:
    """Identifies high-level user intents that require multi-tool coordination."""
    q = query.lower()

    if any(k in q for k in ["good night", "bedtime", "going to bed", "sleep time", "going to sleep"]):
        return MACRO_ROUTINES["bedtime"]

    if any(k in q for k in ["leaving home", "leaving the house", "leave home", "away mode", "going out", "heading out"]):
        return MACRO_ROUTINES["leave_home"]

    if any(k in q for k in ["movie mode", "movie time", "watch movie", "cinema mode"]):
        return MACRO_ROUTINES["movie_mode"]

    if any(k in q for k in ["good morning", "wake up", "morning routine", "waking up"]):
        return MACRO_ROUTINES["morning"]

    if any(k in q for k in ["emergency", "fire alarm", "intruder", "panic mode"]):
        return MACRO_ROUTINES["emergency"]

    return None


def decompose_compound_query(query: str) -> List[str]:
    """
    Decomposes compound sentences joined by conjunctions into distinct sub-actions.
    e.g. 'turn on kitchen lights, set gpio 14 to high and lock front door'
    -> ['turn on kitchen lights', 'set gpio 14 to high', 'lock front door']
    """
    # Check for explicit multi-action delimiters
    # Replace separators with a common delimiter '|'
    cleaned = re.sub(r",\s*(?:and|then)\s+", "|", query, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:and\s+then|then|after\s+that)\s+", "|", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+and\s+(?=(?:turn|set|switch|dim|lock|unlock|open|close|read|check|drive|write|gpio|pwm)\b)", "|", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r";\s*", "|", cleaned)

    parts = [p.strip() for p in cleaned.split("|") if p.strip()]
    return parts if len(parts) > 1 else [query.strip()]


class ToolPlanner:
    """Orchestrates multi-step tool planning and execution via the neural model."""
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def plan_tools(self, query: str) -> Dict[str, Any]:
        """
        Creates an actionable, structured tool execution plan for the given user request.
        """
        # 1. Check if the query triggers a high-level macro routine
        macro = detect_macro_intent(query)
        if macro:
            return {
                "is_macro": True,
                "goal": macro["goal"],
                "description": macro["description"],
                "strategy": "sequential",
                "sub_queries": [s["query"] for s in macro["steps"]],
                "reasons": [s["reason"] for s in macro["steps"]],
            }

        # 2. Check for compound sentences
        sub_queries = decompose_compound_query(query)
        if len(sub_queries) > 1:
            return {
                "is_macro": False,
                "goal": f"Multi-Tool Compound Plan ({len(sub_queries)} Steps)",
                "description": "Decomposed compound request into sequential hardware & device operations",
                "strategy": "sequential",
                "sub_queries": sub_queries,
                "reasons": [f"Execute: '{sq}'" for sq in sub_queries],
            }

        # 3. Single-step query
        return {
            "is_macro": False,
            "goal": "Direct Tool Execution",
            "description": "Single-step tool operation",
            "strategy": "direct",
            "sub_queries": [query],
            "reasons": ["Direct user intent"],
        }
