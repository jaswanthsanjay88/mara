"""
Deploys the in-browser WebAssembly Mara AFM site to Hugging Face Spaces:
https://huggingface.co/spaces/jaswanthsanjay88/mara-demo
"""

import os
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
SPACE_ID = "jaswanthsanjay88/mara-demo"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_TO_UPLOAD = [
    ("space_static/index.html", "index.html"),
    ("space_static/README.md", "README.md"),
]

def main():
    print(f"Deploying in-browser WASM Mara AFM to Space: {SPACE_ID}...")
    api = HfApi(token=TOKEN)

    # Ensure space exists with static SDK
    try:
        api.create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            space_sdk="static",
            exist_ok=True,
        )
    except Exception as e:
        print("Space repo verified / created:", e)

    for local_rel, remote_name in FILES_TO_UPLOAD:
        local_path = os.path.join(ROOT, local_rel)
        if not os.path.exists(local_path):
            print(f"[-] Missing {local_path}")
            continue
        print(f"[*] Uploading {local_rel} -> {remote_name}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_name,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message="Deploy in-browser WebAssembly Mara AFM engine downloading from jaswanthsanjay88/mara",
        )
        print(f"    -> Successfully deployed {remote_name}!")

    print("\n" + "=" * 80)
    print(f"[+] Site is now LIVE at: https://huggingface.co/spaces/{SPACE_ID}")
    print(f"[+] Direct Webview:      https://jaswanthsanjay88-mara-demo.hf.space")
    print("=" * 80)

if __name__ == "__main__":
    main()
