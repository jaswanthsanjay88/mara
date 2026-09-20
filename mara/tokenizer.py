"""
Mara Tokenizer & Sequence Encoders.
Supports native prefill-only decision delimiters:
<|endoftext|>, <|state|>, <|q|>, <|opt|>, </opt>, <|decide|>
"""

import re
from typing import Any
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

VOCAB_SIZE = 8192

SPECIAL_TOKENS = [
    "<|endoftext|>",  # 0
    "<|state|>",      # 1
    "<|q|>",          # 2
    "<|opt|>",        # 3
    "</opt>",         # 4
    "<|decide|>",     # 5
]

SPECIAL_MAP = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

OPT_NONE, OPT_DECIDE = -1, -2


def train_tokenizer(texts, out_path: str, vocab_size: int = VOCAB_SIZE) -> PreTrainedTokenizerFast:
    """Trains a byte-level BPE tokenizer with reserved decision delimiters."""
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    tok.save(out_path)
    return load_tokenizer(out_path)


def load_tokenizer(path: str) -> PreTrainedTokenizerFast:
    """Loads tokenizer from JSON file."""
    hf = PreTrainedTokenizerFast(
        tokenizer_file=path,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL_TOKENS[1:],
    )
    return hf


def user_tokens(tok: PreTrainedTokenizerFast, text: str) -> list[int]:
    """
    Tokenizes user-supplied text safely so user inputs cannot spoof control tokens.
    Rewrites `<|name|>` to `<¦name¦>` before tokenization.
    """
    sanitized = _SPECIAL_RE.sub(r"<¦\1¦>", str(text))
    return tok(sanitized, add_special_tokens=False).input_ids


def encode_record(
    tok: PreTrainedTokenizerFast,
    rec: dict[str, Any],
    max_state: int = 384,
    max_branch: int = 1024,
) -> dict[str, Any]:
    """
    Packs a state document + multiple question branches into a single token sequence.
    Branch position IDs restart right after state so question order has zero positional bias.
    """
    state_toks = user_tokens(tok, rec.get("state", ""))
    s_id = tok.convert_tokens_to_ids("<|state|>")
    q_id = tok.convert_tokens_to_ids("<|q|>")
    o_id = tok.convert_tokens_to_ids("<|opt|>")
    c_id = tok.convert_tokens_to_ids("</opt>")
    d_id = tok.convert_tokens_to_ids("<|decide|>")

    S = [s_id] + state_toks[: max_state - 1]
    ids = list(S)
    seg = [0] * len(S)
    pos = list(range(len(S)))
    opt = [OPT_NONE] * len(S)

    decide_idx: list[int] = []
    opt_idx: list[list[int]] = []
    p0 = len(S)

    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, str(o)) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]

        if len(br) > max_branch:
            br = br[:max_branch - 1] + [d_id]

        base = len(ids)
        br_pos = list(range(p0, p0 + len(br)))
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        br_opt = br_opt[:len(br)]

        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)

        ids += br
        seg += [k] * len(br)
        pos += br_pos
        opt += br_opt
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + e for e in ends])

    labels = [q.get("label", 0) for q in rec["questions"]]

    return {
        "ids": ids,
        "seg": seg,
        "pos": pos,
        "opt": opt,
        "decide_idx": decide_idx,
        "opt_idx": opt_idx,
        "labels": labels,
    }


def encode_state(tok: PreTrainedTokenizerFast, state_text: str, max_state: int = 384) -> dict[str, Any]:
    """Encodes the state prefix for cold prefill."""
    s_id = tok.convert_tokens_to_ids("<|state|>")
    state_toks = user_tokens(tok, state_text)
    prefix_ids = [s_id] + state_toks[: max_state - 1]
    return {
        "ids": prefix_ids,
        "pos": list(range(len(prefix_ids))),
        "length": len(prefix_ids),
    }


def encode_question_branches(
    tok: PreTrainedTokenizerFast,
    questions: list[dict[str, Any]],
    state_len: int,
) -> dict[str, Any]:
    """Encodes question branches to evaluate against a cached state prefix."""
    q_id = tok.convert_tokens_to_ids("<|q|>")
    o_id = tok.convert_tokens_to_ids("<|opt|>")
    c_id = tok.convert_tokens_to_ids("</opt>")
    d_id = tok.convert_tokens_to_ids("<|decide|>")
    branches = []

    for q in questions:
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, str(o)) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        br_pos = list(range(state_len, state_len + len(br)))

        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)

        branches.append({
            "ids": br,
            "pos": br_pos,
            "decide_idx": len(br) - 1,
            "opt_idx": ends,
            "num_opts": len(spans),
        })

    max_len = max(len(b["ids"]) for b in branches) if branches else 0
    return {"branches": branches, "max_len": max_len}
