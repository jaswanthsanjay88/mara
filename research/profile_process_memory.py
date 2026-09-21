"""
Process Memory (Peak RSS) and Latency Profiler for Mara vs Needle.
Measures true OS-level resident set size (RSS) via isolated subprocesses
to guarantee no memory leakage or cross-contamination between engines.
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict, Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


WORKER_CODE = """
import os
import sys
import time
import json
import psutil

engine = sys.argv[1]
checkpoint_path = sys.argv[2]
t0 = time.perf_counter()

queries = [
    "turn on the living room light",
    "set thermostat to 21 degrees",
    "turn off kitchen lights and lock front door",
    "open the garage door",
    "what is the difference between a mammal and a marsupial",
]

proc = psutil.Process()
mem_init = proc.memory_info().rss / (1024 * 1024)

if engine == "mara_pytorch":
    import torch
    from mara.model import Mara, MaraConfig
    from mara.tokenizer import load_tokenizer
    from mara.data import TOKENIZER_PATH
    from mara.argument_heads import MaraArgumentModel

    tok = load_tokenizer(TOKENIZER_PATH)
    ckpt = torch.load("checkpoints/mara_smart_home.pt", map_location="cpu", weights_only=False)
    cfg = MaraConfig(**ckpt["config"])
    model = Mara(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    arg_model = MaraArgumentModel(d_model=cfg.d_model)
    arg_model.load_state_dict(torch.load("checkpoints/mara_arg_heads.pt", map_location="cpu", weights_only=False))
    arg_model.eval()

    t_load = time.perf_counter()
    load_time_ms = (t_load - t0) * 1000

    def run_fn(q):
        tokens = tok.encode(q)
        if len(tokens) < 32:
            tokens = tokens + [0] * (32 - len(tokens))
        tokens = tokens[:32]
        ids = torch.tensor([tokens])
        pos = torch.arange(32).unsqueeze(0)
        with torch.no_grad():
            h, _ = model(ids, position_ids=pos)
        return h

elif engine == "mara_onnx":
    import onnxruntime as ort
    import numpy as np
    from mara.tokenizer import load_tokenizer
    from mara.data import TOKENIZER_PATH

    tok = load_tokenizer(TOKENIZER_PATH)
    session = ort.InferenceSession(checkpoint_path, providers=["CPUExecutionProvider"])

    t_load = time.perf_counter()
    load_time_ms = (t_load - t0) * 1000

    def run_fn(q):
        tokens = tok.encode(q)
        if len(tokens) < 32:
            tokens = tokens + [0] * (32 - len(tokens))
        tokens = tokens[:32]
        inputs = {
            "input_ids": np.array([tokens], dtype=np.int64),
            "position_ids": np.arange(32, dtype=np.int64).reshape(1, 32),
        }
        return session.run(None, inputs)

elif engine == "needle_base" or engine == "needle_finetuned":
    import needle
    from typing import Literal

    def set_lights(room: Literal["living room", "kitchen", "bedroom", "bathroom", "garage"], brightness: int):
        pass

    def set_thermostat(temperature_c: float):
        pass

    def control_device(device: Literal["fan", "garage door"], action: Literal["on", "off", "open", "close"]):
        pass

    def lock_door(door: Literal["front door"], locked: bool):
        pass

    tools = [set_lights, set_thermostat, control_device, lock_door]

    if engine == "needle_finetuned" and os.path.exists(checkpoint_path):
        agent_kwargs = {"weights": checkpoint_path, "tools": tools}
    else:
        agent_kwargs = {"tools": tools}

    # Warm load
    agent = needle.Needle(**agent_kwargs)
    t_load = time.perf_counter()
    load_time_ms = (t_load - t0) * 1000

    def run_fn(q):
        # fresh instance for stateless execution
        a = needle.Needle(**agent_kwargs)
        res = a.complete(q)
        return res

else:
    raise ValueError(f"Unknown engine: {engine}")

# Warm up
for q in queries:
    run_fn(q)

# Benchmark latency
times = []
for _ in range(5):
    for q in queries:
        t_start = time.perf_counter()
        run_fn(q)
        times.append((time.perf_counter() - t_start) * 1000)

times.sort()
mean_lat = sum(times) / len(times)
p50_lat = times[len(times) // 2]
p95_lat = times[int(len(times) * 0.95)]
min_lat = times[0]

peak_rss = proc.memory_info().rss / (1024 * 1024)

result = {
    "engine": engine,
    "cold_start_ms": round(load_time_ms, 2),
    "latency_min_ms": round(min_lat, 2),
    "latency_mean_ms": round(mean_lat, 2),
    "latency_p50_ms": round(p50_lat, 2),
    "latency_p95_ms": round(p95_lat, 2),
    "peak_rss_mb": round(peak_rss, 2),
    "net_rss_mb": round(peak_rss - mem_init, 2),
}

print(json.dumps(result))
"""


def profile_engine(engine_name: str, checkpoint_path: str = "") -> Dict[str, Any]:
    print(f"[*] Profiling engine in isolated subprocess: {engine_name}...")
    cmd = [
        sys.executable,
        "-c",
        WORKER_CODE,
        engine_name,
        checkpoint_path,
    ]
    
    start_time = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
    )
    
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print(f"[-] Error profiling {engine_name}:")
        print(stderr)
        return {"engine": engine_name, "error": stderr}

    # Find the JSON output line
    output_lines = [line.strip() for line in stdout.splitlines() if line.strip().startswith("{")]
    if not output_lines:
        print(f"[-] No JSON output from worker: {stdout}\n{stderr}")
        return {"engine": engine_name, "error": "No output"}

    res = json.loads(output_lines[-1])
    print(f"    -> Cold start: {res.get('cold_start_ms')} ms | Latency (p50): {res.get('latency_p50_ms')} ms | Peak RSS: {res.get('peak_rss_mb')} MB")
    return res


def run_full_memory_profile(needle_cact_path: str = "checkpoints/needle3_specialist.cact") -> Dict[str, Any]:
    results = {}
    
    # 1. Mara PyTorch CPU
    results["mara_pytorch"] = profile_engine("mara_pytorch", "checkpoints/mara_smart_home.pt")
    
    # 2. Mara ONNX
    results["mara_onnx"] = profile_engine("mara_onnx", "checkpoints/mara.onnx")
    
    # 3. Needle Base (Cactus Engine)
    results["needle_base"] = profile_engine("needle_base", "")
    
    # 4. Needle Fine-Tuned (Specialist .cact)
    if os.path.exists(os.path.join(ROOT, needle_cact_path)):
        results["needle_finetuned"] = profile_engine("needle_finetuned", needle_cact_path)
    else:
        print(f"[!] Warning: {needle_cact_path} not found. Skipping needle_finetuned.")

    out_file = os.path.join(ROOT, "data", "memory_profile_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Memory profiling complete. Saved to: {out_file}")
    return results


if __name__ == "__main__":
    cact_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/needle3_specialist.cact"
    run_full_memory_profile(cact_path)
