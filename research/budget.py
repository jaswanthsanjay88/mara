import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mara.model import Mara, MaraConfig

TARGET_PARAMS = 24_648_192


def count_params(**cfg_kwargs) -> int:
    return Mara(MaraConfig(**cfg_kwargs)).num_params()


def flops_per_token(cfg: MaraConfig) -> int:
    d = cfg.d_model
    n_ffn = sum(1 for i in range(cfg.n_layers) if cfg.ffn_every != 0 and i % cfg.ffn_every == 0)
    if cfg.ffn_bottleneck:
        hidden = cfg.ffn_bottleneck
    else:
        hidden = ((int(d * cfg.ffn_hidden_mult) + 255) // 256) * 256
    attn = cfg.n_layers * cfg.n_loops * (12 * d * d)
    ffn = n_ffn * cfg.n_loops * (6 * d * hidden)
    embed_lookup = 0
    lm_head = 2 * d * cfg.vocab_size
    return attn + ffn + lm_head + embed_lookup


def match_depth(base_kwargs: dict, target: int = TARGET_PARAMS, max_depth: int = 64) -> dict:
    best = None
    for depth in range(1, max_depth + 1):
        kwargs = {**base_kwargs, "n_layers": depth}
        n = count_params(**kwargs)
        diff = abs(n - target)
        if best is None or diff < best[1]:
            best = (kwargs, diff, n)
        if n > target * 1.02:
            break
    kwargs, _, n = best
    print(f"depth {kwargs['n_layers']:2d} -> {n:,} params ({(n-target)/target*100:+.2f}% vs target)")
    return kwargs


def ffn_block_count(kwargs: dict) -> int:
    if kwargs.get("ffn_every", 1) == 0:
        return 0
    return sum(1 for i in range(kwargs["n_layers"]) if i % kwargs.get("ffn_every", 1) == 0)


def match_width(kwargs: dict) -> dict:
    """Depth steps are coarser than the ±1% budget window (one attention-only
    block is ~4% of target), so close the residual with FFN width instead."""
    n = count_params(**kwargs)
    n_ffn = ffn_block_count(kwargs)
    gap = TARGET_PARAMS - n
    if gap == 0 or n_ffn == 0:
        return kwargs
    cfg = MaraConfig(**kwargs)
    if cfg.ffn_bottleneck:
        h_cur = cfg.ffn_bottleneck
    else:
        h_cur = ((int(cfg.d_model * cfg.ffn_hidden_mult) + 255) // 256) * 256
    h_new = max(64, round((h_cur + gap / (3 * cfg.d_model * n_ffn)) / 8) * 8)
    tuned = {**kwargs, "ffn_bottleneck": h_new}
    n2 = count_params(**tuned)
    if abs(n2 - TARGET_PARAMS) < abs(n - TARGET_PARAMS):
        print(f"       ffn hidden {h_cur} -> {h_new}  "
              f"({n2:,} params, {(n2 - TARGET_PARAMS) / TARGET_PARAMS * 100:+.3f}% vs target)")
        return tuned
    return kwargs


VARIANTS = {
    "standard": {"ffn_every": 1},
    "san": {"ffn_every": 0, "qk_norm": True},
    "interleaved2": {"ffn_every": 2},
    "interleaved3": {"ffn_every": 3},
    "bottleneck128": {"ffn_every": 1, "ffn_bottleneck": 128},
    "bottleneck256": {"ffn_every": 1, "ffn_bottleneck": 256},
    "looped2": {"ffn_every": 1, "n_loops": 2},
    "looped4": {"ffn_every": 1, "n_loops": 4},
}


def resolve(variant: str) -> tuple[MaraConfig, dict]:
    base = VARIANTS[variant]
    matched = match_width(match_depth(base))
    cfg = MaraConfig(**matched)
    stats = {
        "variant": variant,
        "params": count_params(**matched),
        "flops_per_token": flops_per_token(cfg),
        "config": matched,
    }
    return cfg, stats


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--variant", default=None)
    args = p.parse_args()
    names = list(VARIANTS) if args.all else [args.variant]
    for name in names:
        cfg, stats = resolve(name)
        print(f"{name:14s} layers={cfg.n_layers:2d} loops={cfg.n_loops} "
              f"ffn_every={cfg.ffn_every} bottleneck={cfg.ffn_bottleneck} "
              f"| params {stats['params']:,} | GFLOPs/tok {stats['flops_per_token']/1e9:.3f}")
