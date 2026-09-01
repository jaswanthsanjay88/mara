import glob
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
DOMAIN_DIR = os.path.join(ROOT, "data", "domain")
OUT_DIR = os.path.join(ROOT, "checkpoints", "smollm2_lora")
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_domain_dataset():
    files = glob.glob(os.path.join(DOMAIN_DIR, "*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no .jsonl found in {DOMAIN_DIR}. "
            "add files with lines like: "
            '{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}'
        )
    ds = load_dataset("json", data_files=files, split="train")
    ds = ds.train_test_split(test_size=0.05, seed=42)
    return ds["train"], ds["test"]


def main(epochs: int = 3):
    device = "cuda"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def format_and_tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        enc = tokenizer(text, truncation=True, max_length=512)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def collate(feats):
        batch = tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in feats],
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds, eval_ds = load_domain_dataset()
    train_ds = train_ds.map(format_and_tokenize, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(format_and_tokenize, remove_columns=eval_ds.column_names)
    print(f"train examples: {len(train_ds)}, eval: {len(eval_ds)}")

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        report_to=[],
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(OUT_DIR)
    print(f"adapter saved to {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    main(p.parse_args().epochs)
