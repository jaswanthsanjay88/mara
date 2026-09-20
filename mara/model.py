"""
DecisionMara: From-scratch prefill-only decision foundation model.
Equipped with:
- Grouped-Query Attention (GQA) & Rotary Position Embeddings (RoPE)
- Block-causal branch masking for single-pass multi-question evaluation
- Native Bilinear PointerHead for zero-generation calibrated decision readout
- Prefix KV-cache support for sub-5ms repeated state queries
"""

import math
from dataclasses import dataclass
from typing import Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MaraConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2  # GQA: 4x KV memory reduction on edge/ESP32
    max_seq_len: int = 1024
    dropout: float = 0.0
    ffn_every: int = 1
    ffn_hidden_mult: float = 3.0
    ffn_bottleneck: int | None = None
    n_loops: int = 1
    qk_norm: bool = False
    pointer_dim: int = 256


class Rotary(nn.Module):
    """Rotary Position Embedding supporting arbitrary position IDs."""
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, heads, head_dim]
        if position_ids is None:
            T = x.size(1)
            cos = self.cos[:T].unsqueeze(0).unsqueeze(2)  # [1, T, 1, head_dim]
            sin = self.sin[:T].unsqueeze(0).unsqueeze(2)
        else:
            # position_ids: [B, T]
            cos = self.cos[position_ids].unsqueeze(2)      # [B, T, 1, head_dim]
            sin = self.sin[position_ids].unsqueeze(2)
        x1, x2 = x.float().chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x.float() * cos + rotated * sin).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats KV heads to match query heads in Grouped-Query Attention (GQA)."""
    if n_rep == 1:
        return x
    B, T, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(B, T, n_kv_heads, n_rep, head_dim)
        .reshape(B, T, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, cfg: MaraConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.rope = Rotary(self.head_dim, cfg.max_seq_len)
        self.q_norm = nn.RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if cfg.qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        if past_kv is not None:
            k_past, v_past = past_kv
            k = torch.cat([k_past, k], dim=1)
            v = torch.cat([v_past, v], dim=1)

        present_kv = (k, v) if use_cache else None

        # Repeat KV heads for GQA if needed
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # Transpose to [B, heads, T, head_dim]
        q = q.transpose(1, 2)
        k_rep = k_rep.transpose(1, 2)
        v_rep = v_rep.transpose(1, 2)

        if attn_mask is None:
            if q.size(2) == k_rep.size(2):
                is_causal = True
            elif past_kv is not None:
                T_q = q.size(2)
                T_k = k_rep.size(2)
                T_past = T_k - T_q
                past_mask = torch.ones((T_q, T_past), dtype=torch.bool, device=q.device)
                causal_mask = torch.tril(torch.ones((T_q, T_q), dtype=torch.bool, device=q.device))
                allow = torch.cat([past_mask, causal_mask], dim=-1).unsqueeze(0).unsqueeze(0)
                min_val = -1e4 if q.dtype in (torch.float16, torch.bfloat16) else -1e9
                attn_mask = torch.zeros((1, 1, T_q, T_k), dtype=q.dtype, device=q.device).masked_fill(~allow, min_val)
                is_causal = False
            else:
                is_causal = False
        else:
            is_causal = False

        drop_p = self.dropout if self.training else 0.0

        y = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            attn_mask=attn_mask,
            dropout_p=drop_p,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y), present_kv


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

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present_kv = self.attn(
            self.norm1(x),
            position_ids=position_ids,
            attn_mask=attn_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + attn_out
        if self.use_ffn:
            x = x + self.mlp(self.norm2(x))
        return x, present_kv


class PointerHead(nn.Module):
    """
    Bilinear pointer readout head scoring criteria options against <decide>.
    logits_j = (k_j @ q) / sqrt(dp)
    """
    def __init__(self, d: int, dp: int = 256):
        super().__init__()
        self.q = nn.Linear(d, dp, bias=False)
        self.k = nn.Linear(d, dp, bias=False)
        self.scale = 1.0 / math.sqrt(dp)

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        # h_decide: [d], h_opts: [K, d] -> logits [K]
        q = self.q(h_decide)
        k = self.k(h_opts)
        return (k @ q) * self.scale


def branch_mask_batch(
    segs: List[List[int]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Additive block-causal attention mask:
    Token i can attend to token j iff:
      1. j <= i (causal)
      2. seg[j] == 0 (state is visible to all) OR seg[j] == seg[i] (same question branch)
    Question branches can NEVER attend to each other.
    """
    B = len(segs)
    L = max(len(s) for s in segs)
    s = torch.full((B, L), -1, device=device, dtype=torch.long)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device, dtype=torch.long)

    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = (causal[None] & same & valid_key) | torch.eye(L, dtype=torch.bool, device=device)[None]

    mask = torch.zeros((B, 1, L, L), dtype=dtype, device=device)
    min_val = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9
    return mask.masked_fill(~allow.unsqueeze(1), min_val)


class Mara(nn.Module):
    """
    Decision Foundation Model:
    Jointly supports next-token language pretraining and single-pass pointer decision readout.
    """
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
        self.lm_head.weight = self.embed.weight  # Weight tying

        self.pointer_head = PointerHead(cfg.d_model, cfg.pointer_dim)

        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self):
        # DeepNorm / residual scaling for deep stack stability
        factor = 1.0 / math.sqrt(2.0 * self.cfg.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.out_proj.weight.mul_(factor)
                if block.use_ffn:
                    block.mlp.w_down.weight.mul_(factor)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = idx.shape
        x = self.embed(idx)
        present_kvs = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            for _ in range(self.cfg.n_loops):
                x, present_kv = block(
                    x,
                    position_ids=position_ids,
                    attn_mask=attn_mask,
                    past_kv=past_kv,
                    use_cache=use_cache,
                )
            if use_cache and present_kv is not None:
                present_kvs.append(present_kv)

        x = self.norm(x)
        return x, present_kvs

    def forward_lm(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Standard causal language modeling forward pass."""
        hidden, _ = self(idx)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def forward_decision(self, packed_batch: dict[str, Any], device: torch.device):
        """
        Evaluates a packed record (state + multiple question branches) under a block-causal mask.
        Computes pointer loss and returns calibrated question probabilities.
        """
        ids = torch.tensor([packed_batch["ids"]], device=device)
        pos = torch.tensor([packed_batch["pos"]], device=device)
        mask = branch_mask_batch([packed_batch["seg"]], device=device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)

        hidden, _ = self(ids, position_ids=pos, attn_mask=mask)
        h = hidden[0]  # [L, d_model]

        probs_out = []
        loss = torch.tensor(0.0, device=device)
        decide_idxs = packed_batch["decide_idx"]
        opt_idxs = packed_batch["opt_idx"]
        labels = packed_batch.get("labels", [])

        for k in range(len(decide_idxs)):
            d_i = decide_idxs[k]
            o_ends = opt_idxs[k]
            h_dec = h[d_i]
            h_opts = h[o_ends]  # [num_opts, d_model]

            logits = self.pointer_head(h_dec, h_opts)  # [num_opts]
            ps = F.softmax(logits, dim=-1)
            probs_out.append(ps.detach().cpu())

            if k < len(labels) and labels[k] >= 0:
                tgt = torch.tensor(labels[k], device=device)
                loss = loss + F.cross_entropy(logits.unsqueeze(0), tgt.unsqueeze(0))

        if len(decide_idxs) > 0:
            loss = loss / len(decide_idxs)

        return probs_out, loss

    def prefill_state(self, prefix_ids: list[int], device: torch.device) -> Tuple[Any, int]:
        """Cold prefill: processes state prefix once and returns prefix KV-cache."""
        ids = torch.tensor([prefix_ids], device=device)
        pos = torch.arange(len(prefix_ids), device=device).unsqueeze(0)
        with torch.no_grad():
            _, kv_cache = self(ids, position_ids=pos, use_cache=True)
        return kv_cache, len(prefix_ids)

    @torch.no_grad()
    def probs_cached(
        self,
        tok,
        rec: dict[str, Any],
        prefix_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        state_len: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Sub-5ms evaluation of questions against a cached state prefix."""
        from .tokenizer import encode_question_branches
        encoded = encode_question_branches(tok, rec["questions"], state_len)
        branches = encoded["branches"]
        if not branches:
            return []

        probs_out = []
        for br in branches:
            br_ids = torch.tensor([br["ids"]], device=device)
            br_pos = torch.tensor([br["pos"]], device=device)

            # Clone prefix cache for isolation
            cached_kv = [(k.clone(), v.clone()) for k, v in prefix_cache]

            hidden, _ = self(
                br_ids,
                position_ids=br_pos,
                past_key_values=cached_kv,
                use_cache=False,
            )
            h = hidden[0]
            d_i = br["decide_idx"]
            o_ends = br["opt_idx"]
            logits = self.pointer_head(h[d_i], h[o_ends])
            probs_out.append(F.softmax(logits, dim=-1).cpu())

        return probs_out

    def probs(self, tok, rec: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
        """Convenience single-pass evaluator for an arbitrary record."""
        from .tokenizer import encode_record
        packed = encode_record(tok, rec)
        probs, _ = self.forward_decision(packed, device=device)
        return probs

    @torch.no_grad()
    def generate_tool_call(
        self,
        tok,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        stop_token: str = "</call>",
    ) -> str:
        """Generates structured tool call tokens until stop_token or max_new_tokens."""
        self.eval()
        stop_id = tok.convert_tokens_to_ids(stop_token)
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            ctx = generated[:, -self.cfg.max_seq_len:]
            hidden, _ = self(ctx)
            logits = self.lm_head(hidden)[:, -1, :] / max(1e-5, temperature)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == stop_id:
                break

        return tok.decode(generated[0].tolist())

    def route_and_execute_tool(
        self,
        tok,
        user_query: str,
        registry,
        device: torch.device,
    ) -> dict[str, Any]:
        """
        Fast-path: Routes user query to registered tools via PointerHead in <5ms,
        and automatically executes the predicted tool in the registry.
        """
        record = registry.compile_fast_path_record(user_query)
        probs = self.probs(tok, record, device=device)
        tool_q_probs = probs[0]
        tool_names = list(registry.tools.keys()) + ["none"]
        best_tool_idx = tool_q_probs.argmax().item()
        chosen_tool = tool_names[best_tool_idx]

        if chosen_tool == "none":
            return {"status": "no_tool_required", "tool": None, "confidence": float(tool_q_probs[best_tool_idx])}

        exec_result = registry.execute({"name": chosen_tool, "arguments": {}})
        return {
            "status": "executed",
            "tool": chosen_tool,
            "confidence": float(tool_q_probs[best_tool_idx]),
            "result": exec_result,
        }

    def configure_optimizers(self, weight_decay=0.1, lr=6e-4, betas=(0.9, 0.95), device_type="cuda"):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = (device_type == "cuda") and hasattr(torch.optim.AdamW, "fused")
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

