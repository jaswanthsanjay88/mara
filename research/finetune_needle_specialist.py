"""
Specialist LoRA Fine-Tuning & Deployment for Needle 3 on Smart Home Benchmark.
Trains Needle 3 LoRA adapter on 1,650 domain examples using native JAX/Optax pipeline,
then exports the specialist archive to `checkpoints/needle3_specialist.cact`.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Enable multi-threaded Eigen for fast CPU training
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=12"
os.environ["PYTHONIOENCODING"] = "utf-8"

from needle.model.finetune import finetune_local, build_main


class FinetuneArgs:
    jsonl_path = os.path.join(ROOT, "data", "needle_train.jsonl")
    checkpoint = os.path.join(ROOT, "checkpoints", "needle3.safetensors")
    epochs = 1
    batch_size = 32
    lr = 2e-4
    lora_rank = 16
    lora_alpha = 32.0
    max_len = 512
    val_split = 0.1
    seed = 42
    generate = 0
    checkpoint_dir = os.path.join(ROOT, "checkpoints")
    out = os.path.join(ROOT, "checkpoints", "needle_specialist_lora.safetensors")


class BuildArgs:
    checkpoint = os.path.join(ROOT, "checkpoints", "needle3.safetensors")
    lora = os.path.join(ROOT, "checkpoints", "needle_specialist_lora.safetensors")
    out = os.path.join(ROOT, "checkpoints", "needle3_specialist.cact")
    platform = None
    layers = None
    upload = False


def main():
    print("=" * 80)
    print("STARTING SPECIALIST LORA FINE-TUNING FOR NEEDLE 3 (1,650 SAMPLES)")
    print("=" * 80)

    t0 = time.time()
    finetune_local(FinetuneArgs())
    print(f"\n[+] Fine-tuning completed in {(time.time() - t0)/60:.1f} minutes.")

    print("\n" + "=" * 80)
    print("BUILDING SPECIALIST CACT ARCHIVE")
    print("=" * 80)
    build_main(BuildArgs())
    print(f"\n[+] Specialist build complete! Output: {BuildArgs.out}")


if __name__ == "__main__":
    main()
