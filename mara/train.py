import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from .model import Mara, MaraConfig
from .tokenizer import load_tokenizer

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.environ.get("MARA_CKPT_DIR", os.path.join(ROOT, "checkpoints"))
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")


@dataclass
class TrainConfig:
    seq_len: int = 512
    batch_size: int = 64
    grad_accum: int = 1
    max_steps: int = 20_000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    eval_interval: int = 500
    log_interval: int = 10
    save_interval: int = 2_000
    fp16: bool = False


def get_batch(data: np.memmap, batch_size: int, seq_len: int, device: str):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, cfg: TrainConfig, device: str) -> dict:
    model.eval()
    out = {}
    for name, path in (("train", TRAIN_BIN), ("val", VAL_BIN)):
        data = np.memmap(path, dtype=np.uint16, mode="r")
        losses = torch.zeros(20)
        for k in range(20):
            x, y = get_batch(data, cfg.batch_size // 2, cfg.seq_len, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def get_lr(it: int, cfg: TrainConfig) -> float:
    if it < cfg.warmup_steps:
        return cfg.lr * (it + 1) / cfg.warmup_steps
    progress = (it - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def main(max_steps: int | None = None, resume: bool = False, fp16: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = TrainConfig()
    if max_steps is not None:
        cfg.max_steps = max_steps
    cfg.fp16 = fp16

    train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_data):,}")

    model_cfg = MaraConfig()
    model = Mara(model_cfg).to(device)
    print(f"parameters: {model.num_params():,}")
    optimizer = model.configure_optimizers(lr=cfg.lr)

    step = 0
    ckpt_path = os.path.join(CKPT_DIR, "mara_small.pt")
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resumed from step {step}")

    model.train()
    amp_dtype = torch.float16 if cfg.fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16)
    t0 = time.time()
    while step < cfg.max_steps:
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for micro in range(cfg.grad_accum):
            x, y = get_batch(train_data, cfg.batch_size, cfg.seq_len, device)
            with torch.autocast(device_type=device, dtype=amp_dtype):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            if cfg.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if cfg.fp16:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if cfg.fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            tok_s = cfg.batch_size * cfg.seq_len * cfg.log_interval / dt
            print(f"step {step:6d} | loss {loss.item() * cfg.grad_accum:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s")
            t0 = time.time()

        if step > 0 and step % cfg.eval_interval == 0:
            losses = estimate_loss(model, cfg, device)
            print(f"[eval] step {step} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        if step > 0 and step % cfg.save_interval == 0 or step == cfg.max_steps - 1:
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "config": vars(model_cfg), "train_config": vars(cfg)}, ckpt_path)
            print(f"saved checkpoint at step {step}")
        step += 1

    generate_demo(model, device)


@torch.no_grad()
def generate_demo(model, device: str):
    from .tokenizer import load_tokenizer
    tok = load_tokenizer(TOKENIZER_PATH)
    prompts = ["Once upon a time", "Lily and Tom went"]
    for p in prompts:
        ids = torch.tensor([tok.encode(p)], device=device)
        out = model.generate(ids, max_new_tokens=150)
        text = tok.decode(out[0].tolist())
        print(f"\n--- prompt: {p!r} ---\n{text}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()
    main(max_steps=args.max_steps, resume=args.resume, fp16=args.fp16)
