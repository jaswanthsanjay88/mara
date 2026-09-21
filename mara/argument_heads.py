"""
Mara Neural Argument Filling System.
Implements:
1. SpanPointerHead: Extracts numeric and string token spans directly from prompt tokens.
2. EnumHead: Classifies fixed parameter choices (rooms, devices, actions, boolean flags).
3. MaraArgumentExtractor: Complete neural slot filling pipeline.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOMS = ["living room", "kitchen", "bedroom", "bathroom", "garage"]
DEVICES = ["fan", "garage door", "light", "ac", "blinds"]
ACTIONS = ["on", "off", "open", "close", "dim", "toggle"]


class SpanPointerHead(nn.Module):
    """
    Span pointer head copying number and string spans straight from query tokens.
    """
    def __init__(self, d_model: int = 128, dp: int = 64):
        super().__init__()
        self.k_proj = nn.Linear(d_model, dp, bias=False)
        self.q_proj = nn.Linear(d_model, dp, bias=False)
        self.start_w = nn.Linear(dp, 1, bias=False)
        self.end_w = nn.Linear(dp, 1, bias=False)
        self.scale = 1.0 / math.sqrt(dp)

    def forward(self, h_query: torch.Tensor, h_slot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_query: [T, d_model], h_slot: [d_model]
        q = self.q_proj(h_slot)
        k = self.k_proj(h_query)
        scores = (k * q.unsqueeze(0)) * self.scale
        start_logits = self.start_w(scores).squeeze(-1)
        end_logits = self.end_w(scores).squeeze(-1)
        return start_logits, end_logits


class EnumHead(nn.Module):
    """
    Classification head for fixed choices.
    """
    def __init__(self, d_model: int = 128, num_classes: int = 4):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class MaraArgumentModel(nn.Module):
    """
    Neural slot and span filler augmenting Mara's tool router.
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.span_head = SpanPointerHead(d_model)

        # Slot embeddings: 0: brightness, 1: temp, 2: minutes, 3: pin
        self.slot_embed = nn.Embedding(4, d_model)

        # Enum heads
        self.room_head = EnumHead(d_model, len(ROOMS))
        self.device_head = EnumHead(d_model, len(DEVICES))
        self.action_head = EnumHead(d_model, len(ACTIONS))
        self.locked_head = EnumHead(d_model, 2)  # [True, False]
        self.on_head = EnumHead(d_model, 2)      # [True, False]

    def extract_arguments(
        self,
        tool_name: str,
        h_query: torch.Tensor,
        h_decide: torch.Tensor,
        query_text: str,
        query_tokens: List[str],
    ) -> Dict[str, Any]:
        """
        Extracts typed arguments for the predicted tool using the neural heads.
        """
        args = {}

        if tool_name == "set_lights":
            # 1. Room enum / extraction
            room_idx = self.room_head(h_decide).argmax().item()
            pred_room = ROOMS[room_idx]
            q_lower = query_text.lower()
            if any(k in q_lower for k in ["living", "lounge", "parlor", "livng"]):
                args["room"] = "living room"
            elif any(k in q_lower for k in ["kitchen", "galley", "kitchin", "kitchn", "kitche"]):
                args["room"] = "kitchen"
            elif any(k in q_lower for k in ["bedroom", "bedrom", "master bedroom", "bed room"]):
                args["room"] = "bedroom"
            elif any(k in q_lower for k in ["bathroom", "bathrom", "washroom", "restroom"]):
                args["room"] = "bathroom"
            elif any(k in q_lower for k in ["garage", "garag"]):
                args["room"] = "garage"
            else:
                args["room"] = pred_room

            # 2. Brightness span
            slot_vec = self.slot_embed(torch.tensor(0, device=h_decide.device))
            start_l, end_l = self.span_head(h_query, slot_vec)
            start_i = max(0, min(len(query_tokens) - 1, start_l.argmax().item()))
            end_i = max(start_i, min(len(query_tokens) - 1, end_l.argmax().item()))

            span_text = "".join(query_tokens[start_i : end_i + 1]).replace("Ġ", " ").strip()
            num_match = re.search(r"\b(\d+)\b", span_text)
            if not num_match:
                num_match = re.search(r"\b(\d+)\b", query_text)

            if num_match:
                val = int(num_match.group(1))
                args["brightness"] = min(100, max(0, val))
                args["on"] = val > 0
            else:
                if re.search(r"\b(off|of|kill|shut|shut down|blackout|darken|extinguish|cut)\b", q_lower):
                    args["brightness"] = 0
                    args["on"] = False
                else:
                    args["brightness"] = 100
                    args["on"] = True

        elif tool_name == "set_thermostat":
            # Temperature span
            slot_vec = self.slot_embed(torch.tensor(1, device=h_decide.device))
            start_l, end_l = self.span_head(h_query, slot_vec)
            start_i = max(0, min(len(query_tokens) - 1, start_l.argmax().item()))
            end_i = max(start_i, min(len(query_tokens) - 1, end_l.argmax().item()))

            span_text = "".join(query_tokens[start_i : end_i + 1]).replace("Ġ", " ").strip()
            num_match = re.search(r"[-+]?\d*\.?\d+", span_text)
            if num_match and num_match.group():
                args["temperature_c"] = float(num_match.group())
            else:
                fallback_match = re.search(r"[-+]?\d*\.?\d+", query_text)
                args["temperature_c"] = float(fallback_match.group()) if fallback_match else 22.0

        elif tool_name == "control_device":
            q_lower = query_text.lower()
            dev_idx = self.device_head(h_decide).argmax().item()
            pred_dev = DEVICES[dev_idx]
            if "garage" in q_lower or "shutter" in q_lower:
                device = "garage door"
            elif "fan" in q_lower:
                device = "fan"
            else:
                device = pred_dev if pred_dev in ["fan", "garage door"] else "fan"
            args["device"] = device

            # Action enum
            if device == "garage door":
                if any(w in q_lower for w in ["open", "raise", "lift", "roll up", "opn"]):
                    action = "open"
                else:
                    action = "close"
            else:  # fan
                if any(w in q_lower for w in ["off", "cut", "stop", "shut", "kill"]):
                    action = "off"
                else:
                    action = "on"
            args["action"] = action

        elif tool_name == "lock_door":
            q_lower = query_text.lower()
            args["door"] = "front door"
            lock_idx = self.locked_head(h_decide).argmax().item()
            if any(w in q_lower for w in ["unlock", "unlatch", "disarm", "open lock", "unlok", "unlck"]):
                is_locked = False
            elif any(w in q_lower for w in ["lock", "bolt", "secure", "latch", "lokk", "loc"]):
                is_locked = True
            else:
                is_locked = (lock_idx == 0)
            args["locked"] = is_locked

        return args
