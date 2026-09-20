import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer, encode_record
from mara.afm import default_registry

device = torch.device("cpu")
tok = load_tokenizer("data/tokenizer.json")
ckpt = torch.load("checkpoints/mara_afm.pt", map_location=device, weights_only=False)
cfg = MaraConfig(**ckpt["config"])
model = Mara(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

query = "Turn off the kitchen light."
rooms = ["kitchen", "living room", "master bedroom", "bathroom", "garage"]
devices = ["light", "fan", "ac", "heater", "lock", "blinds"]
actions = ["turn_on", "turn_off", "dim", "set_temp"]

rec = {
    "state": f"User Request: {query}\nAvailable Functions: gpio_write, gpio_read, pwm_set, control_device, read_sensor, schedule_timer, none",
    "questions": [
        {
            "instr": "Which function should be triggered to fulfill this automation request?",
            "options": [f"tool: {t}" for t in list(default_registry.tools.keys()) + ["none"]],
            "label": -1,
        },
        {
            "instr": "Which room is specified?",
            "options": [f"room: {r}" for r in rooms],
            "label": -1,
        },
        {
            "instr": "Which device is specified?",
            "options": [f"device: {d}" for d in devices],
            "label": -1,
        },
        {
            "instr": "Which action should be executed?",
            "options": [f"action: {a}" for a in actions],
            "label": -1,
        }
    ]
}

packed = encode_record(tok, rec)
with torch.no_grad():
    probs, _ = model.forward_decision(packed, device=device)

tool_names = list(default_registry.tools.keys()) + ["none"]
print("Query:", query)
print("Predicted Tool:", tool_names[probs[0].argmax().item()], f"{probs[0].max().item()*100:.1f}%")
print("Predicted Room:", rooms[probs[1].argmax().item()], f"{probs[1].max().item()*100:.1f}%")
print("Predicted Device:", devices[probs[2].argmax().item()], f"{probs[2].max().item()*100:.1f}%")
print("Predicted Action:", actions[probs[3].argmax().item()], f"{probs[3].max().item()*100:.1f}%")
