import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from .tokenizer import train_tokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
EOS_ID = 0


def prepare(sample_docs: int | None = None):
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")

    if not os.path.exists(TOKENIZER_PATH):
        n_tok_sample = sample_docs or 50_000
        print(f"training BPE tokenizer on {n_tok_sample} documents...")
        texts = (ds["train"][i]["text"] for i in range(n_tok_sample))
        tok = train_tokenizer(texts, TOKENIZER_PATH)
        print(f"vocab size: {tok.vocab_size}")
    else:
        from .tokenizer import load_tokenizer
        tok = load_tokenizer(TOKENIZER_PATH)

    def encode_split(split_name: str, out_path: str):
        split = ds[split_name] if sample_docs is None else ds[split_name].select(range(min(sample_docs, len(ds[split_name]))))
        ids_buffer = []
        with open(out_path, "wb") as f:
            for ex in tqdm(split, desc=f"tokenizing {split_name}"):
                ids = tok.encode(ex["text"] + "\n\n")
                ids_buffer.extend(ids + [EOS_ID])
                while len(ids_buffer) >= 1_000_000:
                    np.asarray(ids_buffer[:1_000_000], dtype=np.uint16).tofile(f)
                    ids_buffer = ids_buffer[1_000_000:]
            if ids_buffer:
                np.asarray(ids_buffer, dtype=np.uint16).tofile(f)

    encode_split("validation", VAL_BIN)
    encode_split("train", TRAIN_BIN)

    n_train = os.path.getsize(TRAIN_BIN) // 2
    n_val = os.path.getsize(VAL_BIN) // 2
    print(f"done. train tokens: {n_train:,}, val tokens: {n_val:,}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample-docs", type=int, default=None, help="limit docs (smoke test)")
    prepare(p.parse_args().sample_docs)
