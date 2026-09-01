import os

from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
user = api.whoami()["name"]
model_id = f"{user}/mara-small"
space_id = f"{user}/mara-demo"

api.upload_file(
    token=token, repo_id=model_id, repo_type="model",
    path_or_fileobj="README_hf.md", path_in_repo="README.md",
)
print("README.md ->", model_id)

for fname in ["index.html", "app.js"]:
    api.upload_file(
        token=token, repo_id=space_id, repo_type="space",
        path_or_fileobj=f"space_static/{fname}", path_in_repo=fname,
    )
    print(f"{fname} -> {space_id}")

print(f"\nMODEL:  https://huggingface.co/{model_id}")
print(f"SPACE:  https://huggingface.co/spaces/{space_id}")
