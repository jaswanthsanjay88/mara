---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: text-classification
tags:
  - agent
  - tool-use
  - mcp
  - function-calling
  - on-device
  - edge-ai
  - onnx
  - web-inference
  - hardware-control
---

# Mara: On-Device Automation Foundation Model (AFM)

## One-Line Summary

Mara is an open-weight, 2.7 MB on-device decision agent designed for embedded developers, roboticists, and web applications that need sub-1ms neural tool calling, zero-syntax-error hardware orchestration, and direct client-side in-browser inference without cloud dependencies.

---

## Model/Agent Description

### Architecture & Foundation
Mara is a prefill-only decision foundation model (692k active parameters, 2 transformer layers, $d_{\text{model}}=128$, 4 Query heads / 2 Key-Value heads with Grouped-Query Attention, and Rotary Position Embeddings). Instead of relying on slow autoregressive token-by-token text generation, Mara incorporates a learned bilinear **PointerHead**:

$$s_i = h_{\text{decide}}^T \cdot W_{\text{pointer}} \cdot h_{\text{opt}_i}$$

This projects latent intent representations directly against candidate tool schemas and parameter slots in a single forward pass, providing calibrated decision probabilities and eliminating JSON syntax failures.

### Agent Loop
The complete agent loop executes on-device or in the browser in four deterministic phases:
1. **Plan**: When presented with compound requests (e.g., `"turn off kitchen lights and lock front door"`) or macro intents (`"good night"`), the multi-step planner decomposes the goal into a dependency-ordered execution graph.
2. **Tool Call**: The model evaluates available tool candidates in parallel using block-causal branch masking. If the top confidence exceeds the activation threshold (`actAt >= 0.70`), the tool call is dispatched immediately. If confidence is between 0.10 and 0.70, it requests user confirmation. Requests without matching tools are cleanly flagged as `refuse`.
3. **Observe**: The runtime dispatches calls against hardware pins (GPIO, PWM), external MCP servers, or device state registers, collecting returned status codes or telemetry.
4. **Respond**: The agent updates state registers and yields structured confirmation back to the calling client or UI console.

### Tools & MCP Integration
Mara includes native support for:
- **Microcontroller Hardware Primitives**: GPIO digital read/write, PWM duty cycles, ADC sensor polling, and I2C/SPI bus reads.
- **Smart Home & Ambient Devices**: Multi-room lighting controls, HVAC climate targets, and perimeter security locks.
- **Model Context Protocol (MCP)**: Dynamic tool discovery and schema negotiation for local agent networks.
- **Client-Side In-Browser Execution**: Runs directly inside modern browsers via WebAssembly and WebGPU with ONNX Runtime Web.

---

## Intended Use and Out-of-Scope Use

### Intended Use
- **In-Browser Web Agents**: Direct client-side tool routing in Next.js/React web apps without hosting backend inference GPUs or Python servers.
- **Embedded & Microcontroller Control**: Offline execution on Raspberry Pi, ESP32-S3 (via ONNX/C++ runtimes), micro-robotics, and edge controllers.
- **Ambient Automation**: Zero-latency, privacy-preserving smart-home orchestration where user voice/text commands never leave the local network.
- **High-Throughput Tool Gatekeeping**: Serving as a sub-1ms local routing filter in front of larger LLM pipelines to handle routine tool calls cheaply without token cost.

### Out-of-Scope Use
- **Open-Ended Text & Essay Generation**: Mara is a decision and tool-routing transformer, not an open-domain conversational chatbot.
- **Unverified High-Voltage Hardware Switching**: Mara should not be connected directly to life-safety or industrial high-voltage actuators without physical interlocks and manual fail-safes.
- **Multi-Turn Open-World Reasoning**: Complex web research or general commonsense trivia outside registered tool schemas.

---

## Quickstart

### Python Quickstart

```bash
pip install torch safetensors huggingface_hub onnxruntime
```

```python
import torch
from huggingface_hub import hf_hub_download
from mara.model import Mara, MaraConfig
from mara.tokenizer import load_tokenizer
from mara.afm import ToolRegistry

# 1. Download weights and config from Hugging Face
weights_path = hf_hub_download(repo_id="jaswanthsanjay88/mara", filename="mara_smart_home.pt")
tok_path = hf_hub_download(repo_id="jaswanthsanjay88/mara", filename="tokenizer.json")

# 2. Instantiate model and load weights
ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
model = Mara(MaraConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.eval()
tok = load_tokenizer(tok_path)

print(f"Mara AFM loaded successfully! Parameters: {model.num_params():,}")
```

### In-Browser JavaScript Quickstart (ONNX Runtime Web)

```javascript
import * as ort from "onnxruntime-web";

// 1. Download ONNX model directly from Hugging Face Hub
const session = await ort.InferenceSession.create(
  "https://huggingface.co/jaswanthsanjay88/mara/resolve/main/mara.onnx",
  { executionProviders: ["wasm"] }
);

// 2. Prepare query inputs (shape [1, 32])
const inputIds = new BigInt64Array(32); // fill with tokenized ids
const positionIds = new BigInt64Array(Array.from({ length: 32 }, (_, i) => BigInt(i)));

const tensorInputs = {
  input_ids: new ort.Tensor("int64", inputIds, [1, 32]),
  position_ids: new ort.Tensor("int64", positionIds, [1, 32]),
};

// 3. Execute in-browser inference in <1ms
const results = await session.run(tensorInputs);
console.log("Hidden states:", results.hidden_states.data);
```

---

## Empirical Benchmark & Trade-Off Analysis

Mara AFM was evaluated in a rigorous, independent head-to-head audit against **Needle 3** (Base, Zero-Adapter True Control, and LoRA Specialist) across a frozen 250-sample independent suite ([`data/frozen_eval_250.jsonl`](https://github.com/jaswanthsanjay88/mara)) with **0 train-test string collisions**.

Scored via standardized **Ordered Exact Match (OEM)** requiring exact tool names, order, call counts, and typed arguments (Wilson 95% confidence intervals reported in brackets):

| Metric | Mara AFM (ONNX / PyTorch) | Needle 3 Base (35 MB Published) | Needle 3 Control (63 MB Zero-Adapter) | Needle 3 Specialist (63 MB LoRA) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Size on Disk** | **0.15 MB** (ONNX) / **2.78 MB** (PT) | 35.34 MB | 63.44 MB | 63.44 MB |
| **Median CPU Latency** | **0.72 ms** (ONNX) / **2.29 ms** (PT) | 1,002.54 ms | 1,187.23 ms | 1,260.69 ms |
| **Speedup vs Needle** | **1,392× faster** | 1.0× (Baseline) | 0.84× | 0.79× |
| **Clean Queries OEM ($n=50$)** | **92.0%** [81.2%, 96.9%] | 74.0% [60.5%, 84.1%] | 84.0% [71.5%, 91.7%] | 84.0% [71.5%, 91.7%] |
| **Colloquial Paraphrases ($n=50$)** | 44.0% [31.2%, 57.7%] | 46.0% [33.0%, 59.6%] | 50.0% [36.6%, 63.4%] | **52.0%** [38.5%, 65.2%] |
| **Typos & ASR Noise ($n=50$)** | **74.0%** [60.5%, 84.1%] | 42.0% [27.6%, 55.8%] | 42.0% [29.4%, 55.8%] | 44.0% [31.2%, 57.7%] |
| **Compound Multi-Step ($n=50$)** | **64.0%** [50.1%, 75.9%] | 34.0% [22.4%, 47.9%] | 58.0% [44.2%, 70.6%] | 58.0% [44.2%, 70.6%] |
| **Hard Negatives ($n=50$)** | **88.0%** [76.2%, 94.4%] | 80.0% [67.0%, 88.8%] | **88.0%** [76.2%, 94.4%] | **88.0%** [76.2%, 94.4%] |
| **Overall OEM Accuracy ($N=250$)** | **72.4%** [66.6%, 77.6%] | 55.2% [49.0%, 61.2%] | 65.6% [59.5%, 71.2%] | 65.2% [59.1%, 70.8%] |

### Honest Technical Trade-Offs

1. **Accuracy Generalization**:
   - On independently authored colloquial paraphrases that Mara was never trained on, Mara drops to **44.0%**, while general-purpose models maintain ~50–52%.
   - Mara is best suited for deterministic, registered tool domains rather than open-ended conversational English.
2. **True Control Finding**:
   - The zero-adapter control archive (`needle3_control.cact`, 63.44 MB) achieves **65.6%**, proving that domain LoRA fine-tuning did not outperform the base model; the earlier observed boost from 55.2% to 62.4% was due to local `write_export` quantization and packaging differences.
3. **Execution Efficiency**:
   - Mara's true edge is extreme efficiency: **0.72 ms** p50 latency (**1,392× faster** than Needle) and a **153 KB** graph footprint, enabling native in-browser and microcontroller deployment.
4. **Guaranteed Syntax**:
   - Mara's pointer and enum heads strictly prevent JSON malformation or non-existent arguments.

---

## Hardware Requirements

| Platform | Memory Footprint | Inference Latency | Support Status |
| :--- | :---: | :---: | :---: |
| **Web Browser (WASM / WebGPU)** | ~15 MB Heap | 0.7 ms – 2.0 ms | Supported (ONNX Web runtime) |
| **x86 / ARM64 CPU** | ~12 MB RAM | 0.7 ms – 2.5 ms | Supported (ONNX / PyTorch) |
| **Raspberry Pi 4 / 5** | ~12 MB RAM | 3.5 ms – 7.0 ms | Supported (ONNX / Python) |
| **Apple Silicon (M-Series)** | ~12 MB RAM | 0.5 ms – 1.2 ms | Supported (ONNX / MPS) |
| **ESP32-S3 (Microcontroller)** | ~2.8 MB SRAM / Flash | 15 ms – 30 ms | Quantized int8 export |

---

## License & Citation

### License
Released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

### Citation
```bibtex
@misc{mara2026afm,
  title={Mara: Automation Foundation Model for Edge Intelligence and In-Browser Tool Calling},
  author={Mara Contributors},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/jaswanthsanjay88/mara}}
}
```
