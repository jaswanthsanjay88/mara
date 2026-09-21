"""
Uploads the Mara AFM model artifacts to Hugging Face Hub (jaswanthsanjay88/mara).
Artifacts:
1. mara.onnx (ONNX model graph for Web / Edge)
2. mara.onnx.data (ONNX external tensor weights)
3. tokenizer.json (BPE tokenizer)
4. config.json (model architecture configuration)
5. mara_smart_home.pt (PyTorch weights)
6. mara_arg_heads.pt (Neural argument slot heads)
7. README.md (Comprehensive Model Card)
"""

import os
import sys
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "jaswanthsanjay88/mara"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_TO_UPLOAD = [
    ("checkpoints/mara.onnx", "mara.onnx"),
    ("checkpoints/mara.onnx.data", "mara.onnx.data"),
    ("data/tokenizer.json", "tokenizer.json"),
    ("config.json", "config.json"),
    ("checkpoints/mara_smart_home.pt", "mara_smart_home.pt"),
    ("checkpoints/mara_arg_heads.pt", "mara_arg_heads.pt"),
    ("README_hf.md", "README.md"),
]


def main():
    print(f"Connecting to Hugging Face Hub for repo: {REPO_ID}...")
    api = HfApi(token=TOKEN)

    # Ensure repo exists
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print(f"[+] Verified repository: https://huggingface.co/{REPO_ID}")

    for local_rel, remote_name in FILES_TO_UPLOAD:
        local_path = os.path.join(ROOT, local_rel)
        if not os.path.exists(local_path):
            print(f"[-] Warning: File {local_path} does not exist, skipping.")
            continue
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"[*] Uploading {local_rel} ({size_mb:.2f} MB) -> {remote_name}...", flush=True)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_name,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Deploy {remote_name} for in-browser ONNX web inference and edge deployment",
        )
        print(f"    -> Successfully uploaded {remote_name}!", flush=True)

    print("\n" + "=" * 80)
    print(f"[+] All Mara model artifacts deployed successfully to: https://huggingface.co/{REPO_ID}")
    print("=" * 80)


if __name__ == "__main__":
    main()
