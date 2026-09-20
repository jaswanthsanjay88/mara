"""
Unit tests for DecisionMara:
- Mathematical branch isolation (packed vs separate)
- Prefix KV-cache accuracy & speed
- Deterministic pointer readout
"""

import math
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mara.model import Mara, MaraConfig
from mara.tokenizer import encode_record, encode_state, encode_question_branches, SPECIAL_TOKENS
from mara.cache import PrefixKVCache


class DummyTokenizer:
    """Fast in-memory mock tokenizer for unit tests without network dependency."""
    def __init__(self):
        self.special = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.vocab = dict(self.special)
        self._next_id = len(SPECIAL_TOKENS)

    def convert_tokens_to_ids(self, tok):
        return self.special[tok]

    def __call__(self, text, add_special_tokens=False):
        words = text.split()
        ids = []
        for w in words:
            if w not in self.vocab:
                self.vocab[w] = self._next_id
                self._next_id += 1
            ids.append(self.vocab[w])
        class Enc:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        return Enc(ids)


def test_model_initialization():
    cfg = MaraConfig(d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=256)
    model = Mara(cfg)
    assert model.num_params() > 0
    assert len(model.blocks) == 2


def test_branch_isolation():
    """
    Branch Isolation Guarantee:
    Evaluating Q1 and Q2 in a packed sequence MUST yield numerically identical
    probabilities to evaluating Q1 and Q2 in complete isolation (tolerance < 1e-5).
    """
    torch.manual_seed(42)
    tok = DummyTokenizer()
    cfg = MaraConfig(d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=512)
    model = Mara(cfg)
    model.eval()

    rec = {
        "state": "The executive base salary is $500,000. Termination severance is 12 months.",
        "questions": [
            {
                "instr": "What is the annual base salary?",
                "options": ["$250,000", "$500,000", "$1,000,000"],
                "label": 1,
            },
            {
                "instr": "How many months of severance are guaranteed?",
                "options": ["6 months", "12 months", "24 months"],
                "label": 1,
            },
        ],
    }

    # 1. Packed evaluation (both questions together)
    packed = encode_record(tok, rec)
    device = torch.device("cpu")
    packed_probs, _ = model.forward_decision(packed, device=device)

    # 2. Separate evaluation (Q1 alone, Q2 alone)
    q1_rec = {"state": rec["state"], "questions": [rec["questions"][0]]}
    q2_rec = {"state": rec["state"], "questions": [rec["questions"][1]]}

    p1_alone, _ = model.forward_decision(encode_record(tok, q1_rec), device=device)
    p2_alone, _ = model.forward_decision(encode_record(tok, q2_rec), device=device)

    # Compare Q1 probabilities
    diff_q1 = (packed_probs[0] - p1_alone[0]).abs().max().item()
    # Compare Q2 probabilities
    diff_q2 = (packed_probs[1] - p2_alone[0]).abs().max().item()

    print(f"Max difference Q1: {diff_q1:.2e}")
    print(f"Max difference Q2: {diff_q2:.2e}")

    assert diff_q1 < 1e-5, f"Branch isolation violated on Q1: {diff_q1}"
    assert diff_q2 < 1e-5, f"Branch isolation violated on Q2: {diff_q2}"


def test_prefix_kv_cache():
    """
    State Prefix KV-Cache Guarantee:
    Evaluating against the prefilled state prefix cache MUST produce probabilities
    identical to cold evaluation.
    """
    torch.manual_seed(42)
    tok = DummyTokenizer()
    cfg = MaraConfig(d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=512)
    model = Mara(cfg)
    model.eval()

    state_text = "The room temperature is 72 degrees. The kitchen light is currently on."
    questions = [
        {"instr": "What is the temperature?", "options": ["68", "72", "75"], "label": 1},
        {"instr": "Is the kitchen light on?", "options": ["yes", "no"], "label": 0},
    ]
    rec = {"state": state_text, "questions": questions}
    device = torch.device("cpu")

    # 1. Cold packed evaluation
    packed = encode_record(tok, rec)
    cold_probs, _ = model.forward_decision(packed, device=device)

    # 2. Prefix KV-cached evaluation
    encoded_state = encode_state(tok, state_text)
    pkv, s_len = model.prefill_state(encoded_state["ids"], device=device)
    cached_probs = model.probs_cached(tok, rec, pkv, s_len, device=device)

    for i in range(len(questions)):
        diff = (cold_probs[i] - cached_probs[i]).abs().max().item()
        print(f"Cold vs Cached diff Q{i+1}: {diff:.2e}")
        assert diff < 1e-5, f"Cache numerical mismatch on Q{i+1}: {diff}"


if __name__ == "__main__":
    test_model_initialization()
    print("test_model_initialization PASSED")
    test_branch_isolation()
    print("test_branch_isolation PASSED")
    test_prefix_kv_cache()
    print("test_prefix_kv_cache PASSED")
