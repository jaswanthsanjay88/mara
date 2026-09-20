"""
Packs mara codebase into mara.tar.gz and updates paste_this_cell.py and mara.ipynb.
"""

import base64
import json
import os
import tarfile

ROOT = os.path.dirname(os.path.dirname(__file__))
TAR_PATH = os.path.join(ROOT, "mara.tar.gz")
PASTE_PATH = os.path.join(ROOT, "paste_this_cell.py")
NOTEBOOK_PATH = os.path.join(ROOT, "mara.ipynb")

EXCLUDE_DIRS = {".git", "venv", "__pycache__", "checkpoints", "runs", ".pytest_cache"}
EXCLUDE_EXTS = {".pyc", ".pt", ".bin", ".tar.gz"}


def should_exclude(tarinfo):
    name = tarinfo.name
    parts = name.replace("\\", "/").split("/")
    if any(p in EXCLUDE_DIRS for p in parts):
        return None
    if any(name.endswith(ext) for ext in EXCLUDE_EXTS):
        return None
    return tarinfo


def pack():
    print(f"Creating {TAR_PATH}...")
    with tarfile.open(TAR_PATH, "w:gz") as tar:
        for item in ["mara", "research", "space", "space_static", "tests", "requirements.txt", "README_hf.md", "RESEARCH_PLAN.md"]:
            p = os.path.join(ROOT, item)
            if os.path.exists(p):
                tar.add(p, arcname=item, filter=should_exclude)

    size = os.path.getsize(TAR_PATH)
    print(f"Packed {size:,} bytes into mara.tar.gz")

    with open(TAR_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    cell_content = (
        "import base64\n"
        "# 1-Click extraction: recreates mara project\n"
        f"data = b'''{b64}'''\n"
        "raw = base64.b64decode(data)\n"
        "with open('/content/mara.tar.gz', 'wb') as f:\n"
        "    f.write(raw)\n"
        "print(f'Wrote {len(raw):,} bytes to /content/mara.tar.gz')\n"
    )

    with open(PASTE_PATH, "w", encoding="utf-8") as f:
        f.write(cell_content)
    print(f"Updated {PASTE_PATH}")

    # Update notebook cell 0 if mara.ipynb exists
    if os.path.exists(NOTEBOOK_PATH):
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)
        if len(nb.get("cells", [])) > 0:
            nb["cells"][0]["source"] = [line + "\n" for line in cell_content.splitlines()]
            # Update cell 1
            if len(nb["cells"]) > 1:
                nb["cells"][1]["source"] = [
                    "%cd /content\n",
                    "!tar xzf mara.tar.gz\n",
                    "!pip -q install \"transformers>=4.44\" datasets tokenizers peft accelerate bitsandbytes\n",
                    "import torch; print(torch.cuda.get_device_name(0), \"| torch\", torch.__version__)\n",
                ]
            with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            print(f"Updated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    pack()
