import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mara.model import Mara
from research.budget import resolve


def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, batch_size, seq_len, device, iters=20):
    model.eval()
    bf16_ok = not hasattr(torch.cuda, "is_bf16_supported") or torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, batch_size // 2, seq_len, device)
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default=None)
    p.add_argument("--max-steps", type=int, default=9_000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--micro-batch", type=int, default=32,
                   help="grad-accumulation micro-batch; must divide batch-size")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    micro = args.micro_batch or args.batch_size
    assert args.batch_size % micro == 0, "--batch-size must be divisible by --micro-batch"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg, stats = resolve(args.variant)
    out_dir = args.out or os.path.join("runs", args.variant)
    os.makedirs(out_dir, exist_ok=True)

    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")

    model = Mara(cfg).to(device)
    optimizer = model.configure_optimizers(lr=args.lr)

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        print(f"[{args.variant}] resumed from ckpt at step {start_step}")

    csv_path = os.path.join(out_dir, "metrics.csv")
    new_file = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    writer = csv.writer(csv_f)
    if new_file:
        writer.writerow(["variant", "step", "loss", "val_loss", "lr", "tok_s"])

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        min_ratio = 0.1
        return args.lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    print(f"[{args.variant}] params {model.num_params():,} | "
          f"GFLOPs/tok {stats['flops_per_token']/1e9:.3f} | steps {args.max_steps} | "
          f"micro-batch {micro} x {args.batch_size // micro}")
    model.train()
    t0, tokens_seen = time.time(), 0
    for step in range(start_step, args.max_steps):
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(args.batch_size // micro):
            x, y = get_batch(train_data, micro, args.seq_len, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if not hasattr(torch.cuda, 'is_bf16_supported') or torch.cuda.is_bf16_supported() else torch.float16):
                _, loss = model(x, y)
            loss.backward()
            loss_val += loss.item() * micro
        loss_val /= args.batch_size
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += args.batch_size * args.seq_len

        if step % 10 == 0:
            dt = time.time() - t0
            tok_s = tokens_seen / dt
            writer.writerow([args.variant, step, f"{loss_val:.4f}", "", f"{lr:.2e}", f"{tok_s:,.0f}"])
            if step % 100 == 0:
                print(f"[{args.variant}] step {step:5d} loss {loss_val:.4f} lr {lr:.2e} {tok_s:,.0f} tok/s")
                csv_f.flush()
                t0, tokens_seen = time.time(), 0

        if (step > 0 and step % args.eval_interval == 0) or step == args.max_steps - 1:
            val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
            vl = eval_loss(model, val_data, args.batch_size, args.seq_len, device)
            writer.writerow([args.variant, step, f"{loss_val:.4f}", f"{vl:.4f}", f"{lr:.2e}", ""])
            csv_f.flush()
            print(f"[{args.variant}] EVAL step {step} val {vl:.4f}")

        if (step + 1) % args.ckpt_every == 0 or step == args.max_steps - 1:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step + 1, "config": vars(cfg), "stats": stats}, ckpt_path)
            print(f"[{args.variant}] checkpoint saved at step {step + 1}")

    torch.save({"model": model.state_dict(), "config": vars(cfg), "stats": stats},
               os.path.join(out_dir, "final.pt"))
    csv_f.close()
    print(f"[{args.variant}] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
