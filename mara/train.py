"""
Training script for DecisionMara: From-scratch base model pretraining.
Jointly trains:
1. Bilinear PointerHead on typed decision criteria (noul, choice, score + RPS loss)
2. Causal Language Model on underlying token representations
"""

import argparse
import json
import math
import os
import random
import time
from typing import Any
import torch
import torch.nn.functional as F

from .model import Mara, MaraConfig, branch_mask_batch
from .tokenizer import encode_record, load_tokenizer

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
RECORDS_JSONL = os.path.join(DATA_DIR, "decision_records.jsonl")
CKPT_DIR = os.path.join(ROOT, "checkpoints")


def question_loss(logits: torch.Tensor, label: int, qtype: str, ord_w: float = 0.5) -> torch.Tensor:
    dev = logits.device
    y = torch.tensor([label], device=dev)
    loss = F.cross_entropy(logits.unsqueeze(0), y)
    if qtype == "score" and ord_w > 0.0 and len(logits) > 2:
        p = F.softmax(logits, dim=-1)
        observed_cdf = (torch.arange(len(p) - 1, device=dev) >= label).to(p.dtype)
        rps = (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
        loss = loss + ord_w * rps
    return loss


def evaluate(model: Mara, tok, records: list[dict[str, Any]], device: torch.device, max_samples: int = 100) -> dict[str, float]:
    model.eval()
    samples = records[:max_samples]
    total_q, correct_q = 0, 0
    choice_tot, choice_corr = 0, 0
    noul_tot, noul_corr = 0, 0
    score_diffs = []
    total_loss = 0.0

    with torch.no_grad():
        for rec in samples:
            enc = encode_record(tok, rec)
            ids = torch.tensor([enc["ids"]], device=device)
            pos = torch.tensor([enc["pos"]], device=device)
            mask = branch_mask_batch([enc["seg"]], device=device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)

            hidden, _ = model(ids, position_ids=pos, attn_mask=mask)
            h = hidden[0]

            for k, q in enumerate(rec["questions"]):
                d_i = enc["decide_idx"][k]
                o_ends = enc["opt_idx"][k]
                logits = model.pointer_head(h[d_i], h[o_ends])
                pred = logits.argmax().item()
                label = q["label"]
                qtype = q.get("qtype", "choice")

                loss_item = question_loss(logits, label, qtype)
                total_loss += loss_item.item()
                total_q += 1
                is_correct = (pred == label)
                correct_q += is_correct

                if qtype == "noul":
                    noul_tot += 1
                    noul_corr += is_correct
                elif qtype == "choice":
                    choice_tot += 1
                    choice_corr += is_correct
                elif qtype == "score":
                    score_diffs.append(abs(pred - label))

    model.train()
    acc = (correct_q / total_q) if total_q > 0 else 0.0
    noul_acc = (noul_corr / noul_tot) if noul_tot > 0 else 0.0
    choice_acc = (choice_corr / choice_tot) if choice_tot > 0 else 0.0
    score_mae = (sum(score_diffs) / len(score_diffs)) if score_diffs else 0.0

    return {
        "val_loss": total_loss / max(1, total_q),
        "val_acc": acc,
        "val_noul_acc": noul_acc,
        "val_choice_acc": choice_acc,
        "val_score_mae": score_mae,
    }


def get_lr(it: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    progress = (it - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lm-weight", type=float, default=0.2, help="Causal LM auxiliary regularization weight")
    parser.add_argument("--out", default=os.path.join(CKPT_DIR, "mara_decision_base.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Base DecisionMara Pretraining")

    if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(RECORDS_JSONL):
        print("Data files not found. Preparing dataset first...")
        from .data import prepare_dataset
        prepare_dataset()

    tok = load_tokenizer(TOKENIZER_PATH)
    print(f"Loaded tokenizer with vocab size: {tok.vocab_size}")

    with open(RECORDS_JSONL, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records):,} records.")

    split = int(len(records) * 0.95)
    train_records = records[:split]
    val_records = records[split:]

    cfg = MaraConfig(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
    )
    model = Mara(cfg).to(device)
    print(f"DecisionMara initialized with {model.num_params():,} parameters ({args.n_layers} layers, d_model={args.d_model}, GQA={args.n_heads}/{args.n_kv_heads})")

    optimizer = model.configure_optimizers(lr=args.lr, device_type=device.type)
    total_steps = (len(train_records) // args.accum) * args.epochs
    step = 0
    t0 = time.time()
    model.train()

    for ep in range(args.epochs):
        random.shuffle(train_records)
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, rec in enumerate(train_records):
            lr = get_lr(step, args.warmup_steps, total_steps, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            enc = encode_record(tok, rec)
            ids = torch.tensor([enc["ids"]], device=device)
            pos = torch.tensor([enc["pos"]], device=device)
            mask = branch_mask_batch([enc["seg"]], device=device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)

            hidden, _ = model(ids, position_ids=pos, attn_mask=mask)
            h = hidden[0]

            loss_rec = torch.tensor(0.0, device=device)
            for k, q in enumerate(rec["questions"]):
                d_i = enc["decide_idx"][k]
                o_ends = enc["opt_idx"][k]
                logits = model.pointer_head(h[d_i], h[o_ends])
                loss_q = question_loss(logits, q["label"], q.get("qtype", "choice"))
                loss_rec = loss_rec + loss_q

            if len(rec["questions"]) > 0:
                loss_rec = loss_rec / len(rec["questions"])

            # Optional auxiliary LM loss over state tokens
            if args.lm_weight > 0.0 and ids.size(1) > 1:
                lm_logits = model.lm_head(hidden[:, :-1])
                lm_targets = ids[:, 1:]
                loss_lm = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), lm_targets.view(-1), ignore_index=0)
                loss_rec = loss_rec + args.lm_weight * loss_lm

            (loss_rec / args.accum).backward()
            epoch_loss += loss_rec.item()

            if (i + 1) % args.accum == 0 or (i + 1) == len(train_records):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % 25 == 0:
                    dt = time.time() - t0
                    print(f"Epoch {ep+1}/{args.epochs} | Step {step}/{total_steps} | Loss {loss_rec.item():.4f} | LR {lr:.2e} | {dt:.1f}s")
                    t0 = time.time()

        val_metrics = evaluate(model, tok, val_records, device=device)
        print(f"\n--- [Eval Epoch {ep+1}] Val Loss: {val_metrics['val_loss']:.4f} | Accuracy: {val_metrics['val_acc']:.1%} | Noul: {val_metrics['val_noul_acc']:.1%} | Choice: {val_metrics['val_choice_acc']:.1%} | Score MAE: {val_metrics['val_score_mae']:.2f} ---\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": vars(cfg),
        "tokenizer": TOKENIZER_PATH,
    }, args.out)
    print(f"Pretrained base model saved to {args.out}")


if __name__ == "__main__":
    main()
