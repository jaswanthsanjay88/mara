import os
import sys
import time
import requests

TOKEN = os.environ.get("HF_TOKEN", "")
URL = "https://huggingface.co/Cactus-Compute/needle3/resolve/main/checkpoints/needle3.safetensors"
DEST = "E:/mara/checkpoints/needle3.safetensors"

headers = {"Authorization": f"Bearer {TOKEN}"}

print(f"Downloading from {URL} to {DEST}...")
response = requests.get(URL, headers=headers, stream=True, timeout=60)
response.raise_for_status()

total_bytes = int(response.headers.get("content-length", 0))
print(f"Total size: {total_bytes / (1024*1024):.2f} MB")

downloaded = 0
t0 = time.time()
with open(DEST, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total_bytes * 100) if total_bytes else 0
            mb = downloaded / (1024 * 1024)
            speed = mb / (time.time() - t0 + 1e-5)
            print(f"\r  {mb:.1f}/{total_bytes/(1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", end="", flush=True)

print(f"\nDownload complete! Final size: {os.path.getsize(DEST) / (1024*1024):.2f} MB")
