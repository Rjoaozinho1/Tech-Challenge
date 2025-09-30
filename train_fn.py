#!/usr/bin/env python3
"""
Tech Challenge – Fine-tuning script (FLAN‑T5) on AmazonTitles-1.3MM (JSON/JSONL)

Now with **PEFT LoRA** (optional) and optional 8‑bit / 4‑bit quantization for lighter training.

What this script does
---------------------
1) Loads `trn.json` or `trn.jsonl` (expects fields: `title`, `content`).
2) Builds Portuguese prompts (título → descrição).
3) (Optional) Baseline generation before training.
4) Fine‑tunes FLAN‑T5 via `Seq2SeqTrainer`, with **LoRA** if enabled.
5) Saves the adapter (and tokenizer). Optionally merge LoRA into the base weights for export.

Dependencies
------------
pip install "transformers>=4.43" "datasets>=2.19" "accelerate>=0.33" "evaluate" "sacrebleu" peft bitsandbytes torch

Notes
-----
- Works with JSON array or JSONL. Your cleaned 86MB `.jsonl` is supported out‑of‑the‑box.
- For low‑VRAM training, use `--peft true --load_in_8bit true` (or `--load_in_4bit true`).
- The script saves the **LoRA adapter** in `--output_dir`. Use `--merge_lora true` to also export merged full weights.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
from typing import List, Dict, Any

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import IntervalStrategy
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# ---------------------------
# Data loading utilities
# ---------------------------

def _read_json_any(path: str) -> List[Dict[str, Any]]:
    """Reads JSON array or JSONL into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.strip().startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON array expected at top level.")
            return data
        else:
            # JSON Lines
            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
            return rows

# ---------------------------
# Text normalization
# ---------------------------

def _norm(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE).strip()
    return text

# ---------------------------
# Prompt templates (PT-BR)
# ---------------------------
PROMPT_TPL = (
    "Pergunta: Qual é a descrição do produto \"{title}\"?\n"
    "Contexto: título = {title}\n"
    "Responda apenas com a descrição."
)

# ---------------------------
# Dataset preparation
# ---------------------------

def make_hf_dataset(rows: List[Dict[str, Any]], title_key: str, content_key: str,
                    train_ratio: float = 0.8, seed: int = 42,
                    subsample: float = 1.0) -> DatasetDict:
    rng = random.Random(seed)
    # Basic filtering + normalization
    cleaned = []
    for r in rows:
        if title_key not in r or content_key not in r:
            continue
        title = _norm(str(r[title_key]))
        content = _norm(str(r[content_key]))
        if not title or not content:
            continue
        cleaned.append({"title": title, "content": content})

    if subsample < 1.0:
        n = max(1, int(len(cleaned) * subsample))
        cleaned = rng.sample(cleaned, n)

    # Shuffle
    rng.shuffle(cleaned)

    # Split
    split = int(len(cleaned) * train_ratio) if len(cleaned) > 1 else 1
    train_rows = cleaned[:split]
    eval_rows = cleaned[split:] if split < len(cleaned) else []

    # Build prompts
    def _to_features(batch):
        titles = batch["title"]
        descs = batch["content"]
        prompts = [PROMPT_TPL.format(title=t) for t in titles]
        return {"prompt": prompts, "target": descs}

    train_ds = Dataset.from_list(train_rows).map(_to_features, batched=True, remove_columns=["title", "content"])
    eval_ds = Dataset.from_list(eval_rows).map(_to_features, batched=True, remove_columns=["title", "content"]) if eval_rows else Dataset.from_list([])

    return DatasetDict({"train": train_ds, "validation": eval_ds})

# ---------------------------
# Tokenization
# ---------------------------

def tokenize_fn(tokenizer, max_source_len: int, max_target_len: int):
    def _tok(batch):
        model_inputs = tokenizer(
            batch["prompt"],
            max_length=max_source_len,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"], max_length=max_target_len, truncation=True, padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return _tok

# ---------------------------
# Metrics (SacreBLEU)
# ---------------------------
bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # Replace -100 with pad token id for decoding
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu_res = bleu.compute(predictions=pred_str, references=[[l] for l in label_str])
    return {"sacrebleu": bleu_res["score"]}

# ---------------------------
# Utils
# ---------------------------

def print_trainable_params(model):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    pct = 100 * trainable / total if total else 0
    print(f"[peft] trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

# ---------------------------
# Baseline generation (pre‑fine‑tuning)
# ---------------------------

def run_baseline_samples(model, tokenizer, ds_valid: Dataset, out_path: str, n: int = 5):
    import torch
    n = min(n, len(ds_valid))
    if n <= 0:
        return
    sample = ds_valid.select(range(n))
    results = []
    for ex in sample:
        inputs = tokenizer(ex["prompt"], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=128)
        out = tokenizer.decode(gen[0], skip_special_tokens=True)
        results.append({"prompt": ex["prompt"], "target": ex["target"], "baseline": out})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[baseline] Saved {len(results)} examples to {out_path}")

# ---------------------------
# Inference helper
# ---------------------------

def answer_question(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    import torch
    inputs = tokenizer(question, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(gen[0], skip_special_tokens=True)

# ---------------------------
# Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tech Challenge – FLAN‑T5 fine‑tune with optional PEFT/LoRA")
    p.add_argument("--data_path", type=str, default="./trn.jsonl", help="Path to trn.json or trn.jsonl (86MB JSONL ok)")
    p.add_argument("--title_key", type=str, default="title")
    p.add_argument("--content_key", type=str, default="content")
    p.add_argument("--model_name", type=str, default="google/flan-t5-base")
    p.add_argument("--output_dir", type=str, default="./models/flan_t5_amazon_lora")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--train_subset", type=float, default=1.0, help="0<subset<=1.0 to iterate faster")
    p.add_argument("--max_source_length", type=int, default=256)
    p.add_argument("--max_target_length", type=int, default=256)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_baseline", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--do_train", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--do_eval", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--infer", type=str, default=None, help="Run a single inference with either base or trained model")
    p.add_argument("--use_trained", type=str, default=None, help="Path to trained adapter or merged model for inference")
    # PEFT / Quantization
    p.add_argument("--peft", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q, k, v, o, wi_0, wi_1, wo", help="Comma‑sep module name fragments for LoRA")
    p.add_argument("--bias", type=str, default="none", choices=["none","all","lora_only"])
    p.add_argument("--task_type", type=str, default="SEQ_2_SEQ_LM")
    p.add_argument("--load_in_8bit", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    p.add_argument("--load_in_4bit", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    p.add_argument("--merge_lora", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False, help="Export merged full weights after training")
    return p.parse_args()


def main():
    global tokenizer  # needed by compute_metrics
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer & model
    print(f"[load] model={args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    bnb_config = None
    if args.load_in_8bit or args.load_in_4bit:
        bnb_kwargs = {}
        if args.load_in_8bit:
            bnb_kwargs["load_in_8bit"] = True
        if args.load_in_4bit:
            bnb_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",                  # precisa ser string quando 4-bit
                "bnb_4bit_compute_dtype": torch.bfloat16       # ou torch.float16 se preferir
            })
        bnb_config = BitsAndBytesConfig(**bnb_kwargs)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto" if bnb_config is not None else None,
    )

    # PEFT LoRA (light training)
    if args.peft:
        if args.load_in_8bit or args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.bias,
            task_type=args.task_type,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        print_trainable_params(model)

    # Fast path: only inference with a trained model
    if args.infer is not None and args.use_trained:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        try:
            # Try loading as PEFT adapter first
            base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
            peft_model = PeftModel.from_pretrained(base, args.use_trained)
            peft_model = peft_model.to("cuda" if torch.cuda.is_available() else "cpu")
            ans = answer_question(peft_model, tokenizer, args.infer)
        except Exception:
            # Fallback: load a fully merged model dir
            m = AutoModelForSeq2SeqLM.from_pretrained(args.use_trained)
            import torch as _t
            m = m.to("cuda" if _t.cuda.is_available() else "cpu")
            ans = answer_question(m, tokenizer, args.infer)
        print("\nPergunta:", args.infer)
        print("Resposta:", ans)
        return

    # Load dataset
    if not os.path.exists(args.data_path):
        print(f"[warn] data_path {args.data_path} not found. Provide --data_path to your JSON/JSONL", file=sys.stderr)
        sys.exit(1)

    rows = _read_json_any(args.data_path)
    dsdict = make_hf_dataset(
        rows, args.title_key, args.content_key,
        train_ratio=args.train_ratio, seed=args.seed, subsample=args.train_subset,
    )

    # Tokenize
    tok = tokenize_fn(tokenizer, args.max_source_length, args.max_target_length)
    tokenized = dsdict.map(tok, batched=True, remove_columns=["prompt", "target"])

    # Data collator & device
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Optional baseline with the *current* (possibly quantized) model
    import torch
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_baseline and len(dsdict["validation"]) > 0:
        run_baseline_samples(model, tokenizer, dsdict["validation"], os.path.join(args.output_dir, "baseline_samples.jsonl"))

    # Training
    if args.do_train:
        train_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            eval_strategy=IntervalStrategy.STEPS if args.do_eval else IntervalStrategy.NO,
            eval_steps=1000,
            save_steps=1000,
            logging_steps=100,
            save_total_limit=2,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            weight_decay=0.0,
            warmup_ratio=0.03,
            num_train_epochs=args.num_train_epochs,
            predict_with_generate=True,
            bf16=torch.cuda.is_available(),
            fp16=False,
            gradient_checkpointing=True,
            report_to=["none"],
            seed=args.seed,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"] if args.do_eval and len(dsdict["validation"])>0 else None,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics if args.do_eval and len(dsdict["validation"])>0 else None,
        )

        trainer.train()

        # Save adapter (or full model if PEFT disabled)
        if args.peft:
            trainer.model.save_pretrained(args.output_dir)
        else:
            trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[done] Saved to {args.output_dir}")

        # Optionally merge LoRA into base and save a standalone model dir
        if args.peft and args.merge_lora:
            print("[merge] Merging LoRA adapter into base weights...")
            base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
            merged = PeftModel.from_pretrained(base, args.output_dir)
            merged = merged.merge_and_unload()
            merged_dir = os.path.join(args.output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"[merge] Full merged model saved to {merged_dir}")

    # Optional example inference with the freshly trained model
    if args.infer is not None:
        ans = answer_question(model, tokenizer, args.infer)
        print("\nPergunta:", args.infer)
        print("Resposta:", ans)


if __name__ == "__main__":
    main()
