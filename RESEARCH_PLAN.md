# MARA Research Plan — Beating the Attention-Only Frontier

**Version:** 1.0 · **Date:** 2026-08-26 · **Status:** Active
**End goal:** A publishable controlled study + the winning architecture ships as `mara-nano` (ESP32-S3 smart-home brain).

---

## 1. The Mission

Build a language model **from scratch** that is *better per parameter* than the current
public frontier for tiny models (Needle 2, 45M params, Simple Attention Network),
then compress and deploy it on a microcontroller that controls a real room.

We are not fine-tuning someone else's model. Every weight, tokenizer, and data
pipeline is ours. We already proved we can do it: `mara-small` (24.6M params)
was pretrained from random init to coherent stories (loss ~1.40) on a free T4.

```
WHERE WE ARE                    WHERE WE'RE GOING
─────────────                   ─────────────────
mara-small 24.6M      ──R&D──►  mara-nano (winning arch)
trained, loss 1.40              best-per-parameter architecture
on HF + live demo                    │
                                     ├─► Phase 2: command fine-tune (mara-home)
                                     ├─► Phase 3: distill to ~4M student
                                     └─► Phase 4/5: int8 → ESP32-S3 firmware
```

---

## 2. Background: What We're Beating

### 2.1 Needle 2 (Cactus Compute, 2026)

- 45M parameters, 14MB binary, ~28MB RAM per session
- Task: tool calling, device use, structured JSON extraction
- Architecture: "Simple Attention Network" (SAN)
- Quantization: 2-bit (CQ2), grammar-constrained decoding
- Trades wins with models 5–70× larger (FunctionGemma 270M, LFM2.5 230M)

### 2.2 The Science Behind It (arXiv:2607.18363)

Their paper, *"A Controlled Study of Attention-Only Transformers,"* proved:

1. FFN layers hold **⅔ of a transformer's non-embedding parameters**
2. Deleting FFN naively costs 0.47 nats (matched depth) / 0.26 nats (matched FLOPs)
3. **Reinvesting the freed parameter budget into deeper attention recovers almost
   everything: only 0.006 nats worse at matched parameters (0.27% of loss)**
4. The remaining deficit localizes to **parametric recall** — knowledge stored in
   weights. Attention-only models are *better* with context in front of them,
   *worse* when facts must come from inside the network
5. QK-normalization keeps deep (48-layer) attention-only stacks trainable

### 2.3 The Gap We Found

The study tested exactly **two points**: 0% FFN and 100% FFN.

```
loss
  │
  │  ●  ← 100% FFN (standard transformer)
  │
  │                    ●  ← 0% FFN (SAN) — only 0.006 nats worse
  │
  ├──┼────────┼────────┼────────┼──► fraction of FFN budget
  0%     25%     50%     75%    100%
         └────── UNEXPLORED ──────┘
```

Nobody has published the interpolation. Their own diagnosis (recall deficit)
suggests the middle might be a sweet spot: *a little* FFN where it counts.

---

## 3. Hypothesis

> **H1:** At matched parameter count, variants with small or sparse FFN capacity
> (interleaved / bottleneck / looped) recover the parametric-recall deficit of
> pure SAN while keeping most of its parameter efficiency — beating BOTH
> endpoints on a combined metric of LM loss + recall accuracy.

> **H2:** On tool-calling accuracy (Needle-style JSON emission), a ≤25M model
> from the winning variant class matches or beats the published 45M SAN
> configuration after task fine-tuning.

---

## 4. The Variant Family

All variants share: d_model=512, 8 heads, RoPE, RMSNorm (pre-norm), tied
embeddings, vocab 8192, context 512. One codebase (`mara/model.py`), knobs only.

### 4.1 standard (baseline — already trained ✅)

```
x ─► [RMSNorm → Attention → +] ─► [RMSNorm → SwiGLU(1536) → +] ─► next block
```
6 layers. This is mara-small. Every experiment compares against it.

### 4.2 san (their recipe)

```
x ─► [RMSNorm → Attention(QK-norm) → +] ─► next block     (no FFN at all)
```
FFN budget reinvested in depth: ~19–20 layers to match 24.6M params.
QK-norm enabled per their finding (deep attention-only stacks need it).

### 4.3 interleaved2 / interleaved3 (our first novel point)

```
block:  1     2     3     4     5     6 ...
      [A+FFN][A  ][A+FFN][A  ][A+FFN][A  ]     ffn_every=2
      [A+FFN][A  ][A  ][A+FFN][A  ][A  ]      ffn_every=3
```
FFN appears every k-th block. Depth auto-matched (~10 for k=2, ~8–9 for k=3).

### 4.4 bottleneck128 / bottleneck256 (width instead of count)

```
x ─► [A] ─► [SwiGLU(hidden=128 or 256)] ─►      every block, tiny FFN
```
Standard FFN hidden is 1536. We shrink it to 128/256 — every layer keeps a
recall organ, but a small one. Depth ~16–17 to match params.

### 4.5 looped2 / looped4 (compute for parameters)

```
same 6-layer stack, each block applied k times (weight-tied recursion):
x ─► [B1 B1] ─► [B2 B2] ─► ... ─► [B6 B6]        n_loops=2
```
Params stay at 24.6M automatically (weights shared); effective depth doubles.
Tests whether *computation depth* substitutes for *parameter depth* on MCU-class
budgets where memory, not FLOPs, is the hard ceiling.

### 4.6 Budget matching (the scientific core)

`research/budget.py` scans depth for each variant until total params hit
**24,648,192 ± 1%**. FLOPs/token are computed analytically and **reported, not
hidden** — deeper SAN variants cost more compute per token; we compare primarily
at matched *params* (their headline protocol) with FLOPs disclosed.

| variant | est. depth | params | relative FLOPs/token |
|---|---|---|---|
| standard | 6 | 24.65M | 1.00× |
| san | ~19 | 24.65M | ~1.6× |
| interleaved2 | ~10 | 24.65M | ~1.1× |
| interleaved3 | ~9 | 24.65M | ~1.0× |
| bottleneck128 | ~17 | 24.65M | ~0.8× |
| bottleneck256 | ~13 | 24.65M | ~0.9× |
| looped2 | 6 | 24.65M | 2.0× |
| looped4 | 6 | 24.65M | 4.0× |

---

## 5. Experimental Protocol

### 5.1 Training

| Control | Value |
|---|---|
| Data | TinyStories stream (469M tokens, identical train.bin for all runs) |
| Tokens per run | 9,000 steps × 32,768 = **~295M tokens** |
| Optimizer | AdamW fused, β=(0.9, 0.95), wd 0.1 (≥2D params only) |
| LR schedule | cosine 6e-4 → 6e-5, 200-step warmup, clip 1.0 |
| Precision | bf16 autocast |
| Seed | 1337 fixed (single-seed v1; seed-pair replication for the final 2 candidates) |
| Hardware | Kaggle T4×2, ~4.5h per run |

### 5.2 Measurements

1. **Val loss** every 500 steps → `runs/<variant>/metrics.csv`
2. **Parametric-recall probe** (`research/factworld.py`):
   - Inject 24 synthetic device-registry facts into training (repeated corpus)
   - After training: closed-book QA ("Which room is the Light-07 in?")
   - Score: exact-answer accuracy over 72 questions
3. **Tool-call benchmark** (`research/toolbench.py`):
   - 2,000 template-generated commands ("dim the kitchen light to 30 percent")
   - Model must emit `{"name": "control_device", "arguments": {...}}`
   - Score: field-exact JSON accuracy
4. **Efficiency:** params, FLOPs/token, and (post-study) int8 size + ESP32 latency

### 5.3 Run Matrix

```
Session 1 (Kaggle ~9h):  standard  + san
Session 2:               interleaved2 + interleaved3
Session 3:               bottleneck128 + bottleneck256
Session 4:               looped2 + looped4
Session 5:               seed-replication of top-2 candidates
```

**Save Version after every session** — `/kaggle/working` is ephemeral.

### 5.4 Decision Rule (pre-registered — no moving goalposts)

Winner must satisfy ALL of:
- Within 0.01 nats of best val loss, **and**
- Highest recall-probe accuracy among that set, **and**
- Tool-call accuracy ≥ 90% after Phase-2 fine-tune

Tie-breaker: lower FLOPs/token wins (matters on ESP32).

---

## 6. From Research to Product

```
WINNING VARIANT (mara-nano arch)
   │
   ├─► Retrain at 300M→1B tokens (scale check: does ranking hold?)
   │
   ├─► PHASE 2 — mara-home: fine-tune on command→JSON corpus
   │            (toolbench data × 50k, synthesized + augmented)
   │
   ├─► PHASE 3 — distill: mara-home (teacher) → 4M student (mara-micro)
   │            student inherits winning arch, vocab shrinks to ~4k
   │
   ├─► PHASE 4 — export: int8 ONNX → llama2.c-style flat binary
   │            grammar-constrained decode: JSON schema compiled to
   │            token mask (Needle's trick, ours in C)
   │
   └─► PHASE 5 — ESP32-S3 firmware: WiFi → text command → local
                inference <300ms → GPIO relays → state reply
```

---

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| SAN unstable at depth 19 even with QK-norm | medium | cap at 16 layers, accept +2% params |
| 295M tokens too few to separate variants | medium | extend top-3 to 600M tokens in Session 5 |
| Recall probe too easy (facts repeated 40×) | high | v2 probe: facts seen once, paraphrased QA |
| Kaggle session kills long runs | medium | checkpoint every 2k steps; resume flag |
| Loop variants just underperform (known result) | high | that IS a publishable finding — negative results count |
| All variants within noise | low–medium | seed replication + extend tokens; if still tied, the paper's finding is "the middle doesn't matter" — still novel |

---

## 8. Paper Skeleton (target: arXiv preprint)

1. Introduction — the unexplored interpolation
2. Related work — SAN (2607.18363), Needle 2, MoE, weight-tied recursion
3. Variants & budget matching methodology
4. Experimental setup (fully reproducible: one repo, one command per run)
5. Results: loss curves, recall probe, toolbench, FLOPs table
6. Analysis: where does FFN capacity matter most?
7. Conclusion: recommended architecture for ≤25M tool-calling models
8. Limitations: single domain corpus, 25M scale only, single-seed except finalists

---

## 9. Current Status

- [x] mara-small trained, published, live demo
- [x] Variant architecture implemented (one model file, config knobs)
- [x] Budget matcher + experiment runner + probes written
- [ ] Session 1 results (standard vs san)
- [ ] Sessions 2–4
- [ ] Probes on all checkpoints
- [ ] Winner declared → mara-nano → Phase 2

*Every claim in the eventual paper traces to a CSV file in `runs/`.*
