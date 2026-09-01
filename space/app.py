import json

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

REPO_ID = "jaswanthsanjay88/mara-small"

torch.set_num_threads(2)


class Rotary(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x):
        T = x.size(1)
        cos = self.cos[:T].view(1, T, 1, -1)
        sin = self.sin[:T].view(1, T, 1, -1)
        x1, x2 = x.float().chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x.float() * cos + rotated * sin).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["d_model"] // cfg["n_heads"]
        self.qkv = nn.Linear(cfg["d_model"], 3 * cfg["d_model"], bias=False)
        self.proj = nn.Linear(cfg["d_model"], cfg["d_model"], bias=False)
        self.rope = Rotary(self.head_dim, cfg["max_seq_len"])

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        q, k, v = self.rope(q), self.rope(k), v
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg["d_model"] * 3)
        hidden = ((hidden + 255) // 256) * 256
        self.w_gate = nn.Linear(cfg["d_model"], hidden, bias=False)
        self.w_up = nn.Linear(cfg["d_model"], hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg["d_model"], bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg["d_model"])
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg["d_model"])
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def build_model(cfg):
    model = nn.Module()
    model.cfg = cfg
    model.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
    model.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg["n_layers"]))
    model.norm = nn.RMSNorm(cfg["d_model"])
    model.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
    model.lm_head.weight = model.embed.weight
    return model


print("loading model...")
cfg = json.load(open(hf_hub_download(REPO_ID, "config.json")))
model = build_model(cfg)
sd = load_file(hf_hub_download(REPO_ID, "model.safetensors"))
model.load_state_dict({k: v.float() for k, v in sd.items()})
model.eval()
tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(REPO_ID, "tokenizer.json"))
print(f"ready: {sum(p.numel() for p in model.parameters()):,} params")


@torch.no_grad()
def generate(prompt, max_tokens, temperature):
    ids = torch.tensor([tok.encode(prompt)])
    for _ in range(int(max_tokens)):
        ctx = ids[:, -cfg["max_seq_len"]:]
        logits = model.embed(ctx)
        for block in model.blocks:
            logits = block(logits)
        logits = model.lm_head(model.norm(logits))[:, -1, :] / temperature
        v, _ = torch.topk(logits, 50)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        ids = torch.cat((ids, torch.multinomial(probs, num_samples=1)), dim=1)
        if ids[0, -1].item() == 0:
            break
    return tok.decode(ids[0].tolist())


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", value="Once upon a time"),
        gr.Slider(50, 400, value=200, step=10, label="Max tokens"),
        gr.Slider(0.3, 1.2, value=0.8, step=0.05, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated story", lines=12),
    title="mara-small",
    description="24.6M parameter GPT trained from scratch on TinyStories.",
    examples=[["Once upon a time"], ["Lily and Tom went"], ["The little dog saw"]],
)

demo.launch()
