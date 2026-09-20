"""
Fine-tuning script for DecisionMara on smart-home device control and domain records.
Fine-tunes the native Mara PointerHead and transformer backbone.
"""

import argparse
import json
import os
import random
import time
import torch
import torch.nn.functional as F

from .model import Mara, MaraConfig, branch_mask_batch
from .data import TOKENIZER_PATH
from .tokenizer import encode_record, load_tokenizer
from .train import question_loss, evaluate, get_lr

ROOT = os.path.dirname(os.path.dirname(__file__))
CKPT_DIR = os.path.join(ROOT, "checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", default=os.path.join(CKPT_DIR, "mara_decision_base.pt"))
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--data-jsonl", default=os.path.join(ROOT, "data", "decision_records.jsonl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--out", default=os.path.join(CKPT_DIR, "mara_home.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Fine-tuning DecisionMara")

    if not os.path.exists(args.base_ckpt):
        print(f"Base checkpoint not found at {args.base_ckpt}. Please run pretraining first: python -m mara.train")
        return

    tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    cfg = MaraConfig(**ckpt["config"])
    model = Mara(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded base model: {model.num_params():,} parameters, {cfg.n_layers} layers.")

    with open(args.data_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records):,} domain records.")

    split = int(len(records) * 0.9)
    train_records = records[:split]
    val_records = records[split:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_records) // args.accum) * args.epochs
    step = 0
    t0 = time.time()
    model.train()

    for ep in range(args.epochs):
        random.shuffle(train_records)
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, rec in enumerate(train_records):
            lr = get_lr(step, 50, total_steps, args.lr, args.min_lr)
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
        print(f"\n--- [Fine-tune Epoch {ep+1}] Val Loss: {val_metrics['val_loss']:.4f} | Acc: {val_metrics['val_acc']:.1%} | Noul: {val_metrics['val_noul_acc']:.1%} | Choice: {val_metrics['val_choice_acc']:.1%} ---\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": vars(cfg),
        "tokenizer": args.tokenizer,
    }, args.out)
    print(f"Fine-tuned model saved to {args.out}")


if __name__ == "__main__":
    main()
