import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MaraConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    max_seq_len: int = 512
    dropout: float = 0.0
    ffn_every: int = 1
    ffn_hidden_mult: float = 3.0
    ffn_bottleneck: int | None = None
    n_loops: int = 1
    qk_norm: bool = False


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        cos = self.cos[:T].view(1, T, 1, -1)
        sin = self.sin[:T].view(1, T, 1, -1)
        x1, x2 = x.float().chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x.float() * cos + rotated * sin).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: MaraConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = Rotary(self.head_dim, cfg.max_seq_len)
        self.q_norm = nn.RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if cfg.qk_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = self.rope(q), self.rope(k), v
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: MaraConfig):
        super().__init__()
        if cfg.ffn_bottleneck:
            hidden = cfg.ffn_bottleneck
        else:
            hidden = int(cfg.d_model * cfg.ffn_hidden_mult)
            hidden = ((hidden + 255) // 256) * 256
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: MaraConfig, use_ffn: bool = True):
        super().__init__()
        self.use_ffn = use_ffn
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        if use_ffn:
            self.norm2 = nn.RMSNorm(cfg.d_model)
            self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        if self.use_ffn:
            x = x + self.mlp(self.norm2(x))
        return x


class Mara(nn.Module):
    def __init__(self, cfg: MaraConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        blocks = []
        for i in range(cfg.n_layers):
            use_ffn = cfg.ffn_every != 0 and (i % cfg.ffn_every == 0)
            blocks.append(Block(cfg, use_ffn=use_ffn))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len
        x = self.embed(idx)
        for block in self.blocks:
            for _ in range(self.cfg.n_loops):
                x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def configure_optimizers(self, weight_decay=0.1, lr=6e-4, betas=(0.9, 0.95), device_type="cuda"):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
