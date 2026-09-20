"""
Unit tests for TypeSafe SystemOne API integration with DecisionMara.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer
from mara.data import TOKENIZER_PATH
from mara.api import SystemOneRequest, to_record, to_answers


def test_systemone_api():
    device = torch.device("cpu")
    tok = load_tokenizer(TOKENIZER_PATH)
    cfg = MaraConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2)
    model = Mara(cfg).to(device)
    model.eval()

    # Create a realistic smart-home control request
    req = SystemOneRequest(
        state="Device Registry Report:\nLocation: kitchen\nDevice: light\nCurrent Level: 75%\nOperational Status: on",
        questions={
            "device_type": {
                "type": "choice",
                "instructions": "Identify the target device",
                "criteria": {"light": "Illumination fixture", "fan": "Ceiling fan", "ac": "Air conditioner"},
            },
            "is_active": {
                "type": "noul",
                "instructions": "Is the device currently powered on?",
                "criteria": {"false": "Off or standby", "true": "Powered on"},
            },
            "intensity": {
                "type": "score",
                "instructions": "Rate current power level",
                "criteria": ["Off (0%)", "Low (25%)", "Medium (50%)", "High (75%)", "Max (100%)"],
            },
        },
    )

    rec, meta = to_record(req)
    assert len(rec["questions"]) == 3

    probs = model.probs(tok, rec, device=device)
    assert len(probs) == 3

    answers = to_answers([p.tolist() for p in probs], meta)

    assert "device_type" in answers
    assert answers["device_type"]["type"] == "choice"
    assert answers["device_type"]["choice"] in ["light", "fan", "ac"]
    assert "confidence" in answers["device_type"]
    assert "probabilities" in answers["device_type"]

    assert "is_active" in answers
    assert answers["is_active"]["type"] == "noul"
    assert 0.0 <= answers["is_active"]["noul"] <= 1.0

    assert "intensity" in answers
    assert answers["intensity"]["type"] == "score"
    assert 0.0 <= answers["intensity"]["score"] <= 4.0

    print("SystemOne API Response:")
    import json
    print(json.dumps(answers, indent=2))
    print("test_systemone_api PASSED!")


if __name__ == "__main__":
    test_systemone_api()
