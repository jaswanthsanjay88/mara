import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer
from research.factworld import build_fact_world, ROOMS, DEVICES, OWNERS
from research.toolbench import gen as gen_toolbench

QA_TEMPLATES = [
    ("Which room is the {code} in?", "room"),
    ("Who owns the {code}?", "owner"),
    ("What kind of device is the {code}?", "kind"),
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


@torch.no_grad()
def gen_text(model, tok, prompt, max_new_tokens, device):
    ids = tok(prompt, return_tensors="pt").input_ids[:, -model.cfg.max_seq_len:].to(device)
    bf16_ok = not hasattr(torch.cuda, "is_bf16_supported") or torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    with torch.autocast(device_type=device, dtype=dtype):
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=1.0, top_k=1)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def check_data(data_bin, tok, windows=200, seq_len=512):
    """Sample train.bin windows and look for device-registry facts."""
    data = np.memmap(data_bin, dtype=np.uint16, mode="r")
    rng = random.Random(0)
    hits = 0
    pattern = re.compile(r"\b(?:Light|Fan|Ac|Heater|Speaker|Tv)-\d{2}\b")
    for _ in range(windows):
        i = rng.randint(0, len(data) - seq_len - 1)
        text = tok.decode([int(t) for t in data[i:i + seq_len]])
        if pattern.search(text):
            hits += 1
    print(f"[data-check] {hits}/{windows} sampled windows contain device-registry facts")
    if hits == 0:
        print("[data-check] facts appear ABSENT from the training stream -> "
              "recall scores will be ~0 regardless of architecture")
    return hits


def recall_probe(model, tok, device):
    _, _, devices = build_fact_world()
    questions = []
    for code, attrs in devices.items():
        for q_t, attr in QA_TEMPLATES:
            questions.append((q_t.format(code=code), str(attrs[attr])))
    correct, rows = 0, []
    for q, ans in questions:
        out = gen_text(model, tok, q, max_new_tokens=8, device=device)
        ok = norm(out).startswith(norm(ans)) or norm(out) == norm(ans)
        correct += ok
        rows.append({"q": q, "expected": ans, "got": out, "correct": bool(ok)})
    acc = correct / len(questions)
    print(f"[recall] {correct}/{len(questions)} = {acc:.1%}")
    return {"accuracy": acc, "n": len(questions), "rows": rows}


def toolbench_probe(model, tok, device, n):
    examples = gen_toolbench(n=n)
    correct, rows = 0, []
    for ex in examples:
        out = gen_text(model, tok, f"Command: {ex['query']}\nOutput JSON: ",
                       max_new_tokens=48, device=device)
        parsed = None
        try:
            start = out.index("{")
            parsed, _ = json.JSONDecoder().raw_decode(out[start:])
        except (ValueError, json.JSONDecodeError):
            pass
        want = ex["answers"][0]
        ok = isinstance(parsed, dict) and parsed.get("name") == want["name"] \
            and parsed.get("arguments") == want["arguments"]
        correct += ok
        rows.append({"query": ex["query"], "expected": want, "got": out, "correct": bool(ok)})
    acc = correct / len(examples)
    print(f"[toolbench] {correct}/{len(examples)} = {acc:.1%}")
    return {"accuracy": acc, "n": len(examples), "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="e.g. runs/standard/final.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--data-bin", default="data/train.bin")
    p.add_argument("--data-check", action="store_true")
    p.add_argument("--skip-recall", action="store_true")
    p.add_argument("--skip-toolbench", action="store_true")
    p.add_argument("--toolbench-n", type=int, default=200)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.tokenizer)

    if args.data_check:
        check_data(args.data_bin, tok)

    ck = torch.load(args.ckpt, map_location=device)
    cfg = MaraConfig(**ck["config"])
    model = Mara(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[probe] loaded {args.ckpt} ({model.num_params():,} params, {cfg.n_layers} layers)")

    results = {"ckpt": args.ckpt, "params": model.num_params()}
    if not args.skip_recall:
        results["recall"] = recall_probe(model, tok, device)
    if not args.skip_toolbench:
        results["toolbench"] = toolbench_probe(model, tok, device, args.toolbench_n)

    out_path = os.path.join(os.path.dirname(args.ckpt), "probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
