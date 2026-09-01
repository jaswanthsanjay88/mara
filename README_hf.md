---
license: apache-2.0
language: en
tags:
  - text-generation
  - gpt
  - tinystories
  - from-scratch
  - small-language-model
  - onnx
datasets:
  - roneneldan/TinyStories
pipeline_tag: text-generation
---

# mara-small

A 24.6M parameter GPT trained **from scratch** (random init → coherent children's stories) on the TinyStories corpus. Designed as the base model of the [mara](https://github.com/) project: a small-model-first family targeting browser inference and microcontrollers.

**Live demo (runs in your browser, no server):** https://huggingface.co/spaces/jaswanthsanjay88/mara-demo

## Architecture

| Config | Value |
|---|---|
| Parameters | 24,648,192 |
| Layers | 6 |
| d_model | 512 |
| Heads | 8 (head_dim 64) |
| FFN | SwiGLU, hidden 1,536 |
| Positional | RoPE |
| Norm | RMSNorm (pre-norm) |
| Context | 512 tokens |
| Vocab | 8,192 custom byte-level BPE (trained on TinyStories) |

## Training

- Data: TinyStories (~469M train tokens, packed uint16 stream)
- Steps: 8,000 · batch 64 × seq 512 = 32,768 tok/step ≈ 262M tokens seen
- Optimizer: AdamW (fused), β=(0.9, 0.95), wd 0.1 on ≥2D params
- Schedule: cosine 6e-4 → 6e-5, 200 warmup steps, grad clip 1.0
- Precision: fp16 autocast (T4)
- Final training loss: ~1.40

## Files

| File | Size | Purpose |
|---|---|---|
| `model.safetensors` | ~50 MB | fp16 PyTorch weights (training/fine-tuning) |
| `model_fp16.onnx` | 50 MB | browser inference via WebGPU |
| `model_int8.onnx` | 27 MB | browser inference via WASM (all browsers) |
| `tokenizer.json` | — | custom 8k BPE (`<|endoftext|>` = id 0) |
| `config.json` | — | architecture hyperparameters |

## Usage (PyTorch)

The architecture is custom (`mara/model.py` in the source repo): RoPE + SwiGLU + tied embeddings. Load weights like so:

```python
import json, torch
from safetensors.torch import load_file
from mara.model import Mara, MaraConfig          # from the mara repo
from mara.tokenizer import load_tokenizer

cfg = MaraConfig(**json.load(open("config.json")))
model = Mara(cfg)
model.load_state_dict({k: v.float() for k, v in load_file("model.safetensors").items()})
model.eval()

tok = load_tokenizer("tokenizer.json")
ids = torch.tensor([tok.encode("Once upon a time")])
print(tok.decode(model.generate(ids, max_new_tokens=200)[0].tolist()))
```

## Limitations

- Children's-story English only — out-of-domain prompts degrade quickly
- Occasional pronoun/gender confusion, typical at this scale
- No instruction-following or world knowledge; it is a pure next-token storyteller
- Context window of 512 tokens

## Roadmap

This model is the teacher for a distilled intent-parsing student that runs fully offline on an ESP32-S3 for smart-home control.
